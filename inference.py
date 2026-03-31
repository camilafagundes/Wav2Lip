from os import listdir, path
import logging
import numpy as np
import scipy, cv2, os, sys, argparse, audio
from pathlib import Path

# PyTorch >= 2.6: checkpoints Wav2Lip exigem weights_only=False (centralizado em utils.torch_compat).
_p = Path(__file__).resolve().parent
while _p != _p.parent:
	if (_p / "utils" / "torch_compat.py").is_file():
		sys.path.insert(0, str(_p))
		break
	_p = _p.parent
else:
	raise ImportError(
		"Não encontrada a pasta avatar-ai (utils/torch_compat.py). "
		"Defina PYTHONPATH para essa pasta ou use o pipeline via gerar_video.py."
	)
from utils.torch_compat import safe_torch_load

from cv2_unicode import imread as imread_unicode
import json, subprocess, random, string
from tqdm import tqdm

logger = logging.getLogger(__name__)

_AVATAR_AI_ROOT = Path(__file__).resolve().parent.parent
_ffmpeg_exe_cache: str | None = None


def _ffmpeg_executable() -> str:
	"""Caminho absoluto do FFmpeg (imageio-ffmpeg ou PATH); fallback 'ffmpeg'."""
	global _ffmpeg_exe_cache
	if _ffmpeg_exe_cache is not None:
		return _ffmpeg_exe_cache
	try:
		root = str(_AVATAR_AI_ROOT)
		if root not in sys.path:
			sys.path.insert(0, root)
		from ffmpeg_util import resolve_ffmpeg_exe

		_ffmpeg_exe_cache = resolve_ffmpeg_exe()
	except Exception:
		logger.warning("ffmpeg_util indisponível; a usar 'ffmpeg' do PATH", exc_info=False)
		_ffmpeg_exe_cache = "ffmpeg"
	return _ffmpeg_exe_cache


def _run_ffmpeg(argv: list[str]) -> None:
	"""Invoca FFmpeg sem shell=True: cada caminho é argumento separado (evita injeção / quebra com espaços)."""
	exe = _ffmpeg_executable()
	cmd = [exe, *argv]
	proc = subprocess.run(
		cmd,
		check=False,
		shell=False,
		capture_output=True,
		text=True,
		encoding="utf-8",
		errors="replace",
	)
	if proc.returncode != 0:
		tail = (proc.stderr or proc.stdout or "")[-3000:]
		logger.error("ffmpeg falhou (código %s): %s", proc.returncode, tail)
		raise RuntimeError(
			f"FFmpeg terminou com código {proc.returncode}. Verifique ficheiros de entrada e instalação do FFmpeg."
		)
from glob import glob
import torch, face_detection
from models import Wav2Lip

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

_face_ext = Path(args.face).suffix.lower()
if os.path.isfile(args.face) and _face_ext in [".jpg", ".jpeg", ".png"]:
	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def _fallback_static_face_box(image):
	"""Último recurso quando S3FD não vê rosto (cartoons, rosto pequeno): região central-superior típica de retrato."""
	h, w = image.shape[:2]
	cx, cy = w // 2, int(h * 0.36)
	side = int(min(w, h) * 0.72)
	x1 = max(0, cx - side // 2)
	x2 = min(w, cx + side // 2)
	y1 = max(0, cy - int(side * 0.45))
	y2 = min(h, cy + int(side * 0.55))
	return (x1, y1, x2, y2)


def _run_face_detections(detector, imgs, batch_size):
	bs = batch_size
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(imgs), bs)):
				predictions.extend(detector.get_detections_for_batch(np.array(imgs[i:i + bs])))
			return predictions, bs
		except RuntimeError:
			if bs == 1:
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			bs //= 2
			print('Recovering from OOM error; New batch size: {}'.format(bs))


def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	predictions, batch_size = _run_face_detections(detector, images, batch_size)

	missing = [i for i, p in enumerate(predictions) if p is None]
	for scale in (2.0, 3.0):
		if not missing:
			break
		sub = [cv2.resize(images[i], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) for i in missing]
		sub_pred, batch_size = _run_face_detections(detector, sub, min(batch_size, max(1, len(sub))))
		inv = 1.0 / scale
		still = []
		for idx, pred in zip(missing, sub_pred):
			if pred is not None:
				x1, y1, x2, y2 = pred
				predictions[idx] = (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv))
			else:
				still.append(idx)
		missing = still

	if missing and args.static:
		print(
			'[Wav2Lip] Detector não encontrou rosto; usando recorte central (comum em desenhos). '
			'Para melhor resultado use foto real, rosto grande e de frente.',
			file=sys.stderr,
		)
		for i in missing:
			predictions[i] = _fallback_static_face_box(images[i])
		missing = []

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			os.makedirs('temp', exist_ok=True)
			cv2.imwrite('temp/faulty_frame.jpg', image)
			raise ValueError(
				'Face not detected! Use vídeo com rosto visível em todos os quadros ou foto com rosto humano de frente. '
				'Ilustrações: tente PNG com o rosto grande e centrado.'
			)

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	# TorchScript vs dict: safe_torch_load escolhe jit.load ou torch.load (PyTorch 2.6+).
	return safe_torch_load(checkpoint_path, map_location=torch.device(device))

def load_model(path):
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)

	# PyTorch 2.11+: torch.load pode devolver ScriptModule sem passar em isinstance(..., ScriptModule).
	# Só tratar como checkpoint em pickle se for dict (incl. OrderedDict com state_dict ou pesos crus).
	if isinstance(checkpoint, dict):
		model = Wav2Lip()
		s = checkpoint.get("state_dict", checkpoint)
		new_s = {k.replace("module.", ""): v for k, v in s.items()}
		model.load_state_dict(new_s)
	elif isinstance(checkpoint, torch.nn.Module):
		model = checkpoint
	else:
		raise TypeError(
			f"Checkpoint não suportado (esperado dict ou nn.Module): {type(checkpoint)!r}"
		)

	model = model.to(torch.device(device))
	return model.eval()

def main():
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	face_ext = Path(args.face).suffix.lower()
	if face_ext in [".jpg", ".jpeg", ".png"]:
		frame0 = imread_unicode(args.face)
		if frame0 is None:
			raise ValueError('Não foi possível ler a imagem (--face). Verifique o caminho (acentos/caminho Windows).')
		full_frames = [frame0]
		fps = args.fps

	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

			if args.rotate:
				frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)

	print ("Number of frames available for inference: "+str(len(full_frames)))

	os.makedirs("temp", exist_ok=True)

	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		audio_src = str(Path(args.audio).expanduser().resolve())
		temp_wav = str(Path("temp") / "temp.wav")
		_run_ffmpeg(["-y", "-i", audio_src, "-strict", "-2", temp_wav])
		args.audio = temp_wav

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)

	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			model = load_model(args.checkpoint_path)
			print ("Model loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(
				os.path.join("temp", "result.avi"),
				cv2.VideoWriter_fourcc(*'DIVX'),
				fps,
				(frame_w, frame_h),
			)

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

	audio_mux = str(Path(args.audio).expanduser().resolve())
	avi_path = str(Path("temp", "result.avi").resolve())
	out_path = str(Path(args.outfile).expanduser().resolve())
	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	_run_ffmpeg(
		[
			"-y",
			"-i",
			audio_mux,
			"-i",
			avi_path,
			"-strict",
			"-2",
			"-q:v",
			"1",
			out_path,
		]
	)

if __name__ == '__main__':
	main()
