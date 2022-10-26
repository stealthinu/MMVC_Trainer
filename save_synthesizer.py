import os
import utils
from models import SynthesizerTrn
from text.symbols import symbols

def main():
  hps = utils.get_hparams(init=False)

  logger = utils.get_logger(hps.model_dir)
  logger.info(hps)

  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.spec_channels,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      hps_data=hps.data,
      **hps.model)

  _, _, _, _ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*[0-9].pth"), net_g, optimizer=None)

  net_g.save_synthesizer(os.path.join(hps.model_dir, "synthesizer.pth"))

if __name__ == "__main__":
  main()
