import torch
import utils
import commons
from models import SynthesizerTrn
from mel_processing import spectrogram_torch
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
import pyopenjtalk

TEXT = "おおおおおお ああああああ うううううう すすすすすす くくくくくく"
TTS_Speaker_ID = 14

CONFIG_PATH = "./configs/train_config_jvs.json"
NET_PATH = "./fine_model/G_500000_jvs_noda.pth"

def mozi2phone(mozi):
    text = pyopenjtalk.g2p(mozi)
    text = "sil " + text + " sil"
    text = text.replace(' ', '-')
    return text

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file(CONFIG_PATH)

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint(NET_PATH, net_g, None)

stn_test = get_text(mozi2phone(TEXT), hps)
sid = torch.LongTensor([TTS_Speaker_ID])

x_test = stn_test.unsqueeze(0)
x_test_lengths = torch.LongTensor([stn_test.size(0)])

# inferと同じ処理
#audio = net_g.infer(x_test, x_test_lengths, sid=sid, noise_scale=.400, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
x, m_p, logs_p, x_mask = net_g.enc_p(x_test, x_test_lengths)

g = net_g.emb_g(sid).unsqueeze(-1) # [b, h, 1]
logw = net_g.dp(x, x_mask, g=g, reverse=True, noise_scale=0.8)
w = torch.exp(logw) * x_mask * 1.0
w_ceil = torch.ceil(w)
y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1)
attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
attn = commons.generate_path(w_ceil, attn_mask)

m_p_spec = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
logs_p_spec = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

z_p = m_p_spec + torch.randn_like(m_p_spec) * torch.exp(logs_p_spec) * .400
z = net_g.flow(z_p, y_mask, g=g, reverse=True)
o = net_g.dec((z * y_mask)[:,:,:], g=g)

audio = o[0][0]
wav = audio.data.cpu().float().numpy()
write("phonemetest.wav", hps.data.sampling_rate, wav)

# train時のforwardと同じ処理して音素推測を表示させる
# o -> spec ->(spec_to_mel_torch)-> mel 
y_spec = spectrogram_torch(audio, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, center=False)
y_spec = torch.squeeze(y_spec, 0)
y_spec_lengths = torch.LongTensor([y_spec.size(0)])
net_g.forward(x_test, x_test_lengths, y_spec, y_spec_lengths, sid=sid)
