#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#             2020   Vincent Brignatz
# Apache 2.0.#

nj=10
fbank_config=conf/fbank.conf
data_in=[in-directory]
data_out=[out-directory]
features_out=features
vad_out=vad
kaldi_root=[Kaldi-path]
help_message="Usage: $0 [options]
Options:
    --nj <nj>      # number of parallel jobs.
    --data-in      # the path of the data input folder
    --data-out     # the path of the data output folder
    --features-out # the path of the feature output folder (can be heavy)
    --kaldi-root   # the full path to the kaldi install folder
Exemple:
     ./feature-extraction.sh --nj 8 --data-in data/train --data-out data/train-no-sil --features-out features/ --kaldi-root kaldi_root
"

# Parse command-line options :
. utils/parse_options.sh

# Create the file path.sh need by all the kaldi scripts :
echo "export KALDI_ROOT=$kaldi_root
export PATH=\$PWD/utils/:\$KALDI_ROOT/tools/openfst/bin:\$KALDI_ROOT/tools/sph2pipe_v2.5:\$PWD:\$PATH
[ ! -f \$KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 \"The standard file \$KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!\" && exit 1
. \$KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C" > path.sh
# source path.sh :
. ./path.sh

# make_fbank  [options] <data-dir> [<log-dir> [<fbank-dir>] ]
# Generates the filter banks
#   --fbank-config         config passed to compute-fbank-feats.
#   --cmd allows           us to choose between multiple type of execution (queue or run)
#   --write-utt2num-frames writes the session lenght in utt2num_frames file.
utils/fix_data_dir.sh ${data_in}
steps/make_fbank.sh --write-utt2num-frames true --fbank-config $fbank_config --nj $nj --cmd "run.pl" ${data_in} exp/make_fbank ${features_out}

# fix_data_dir.sh <data-dir>
# This script helps ensure that the various files in the directory, are correctly sorted and filtered...
utils/fix_data_dir.sh ${data_in}

# compute_vad_decision.sh [options] <data-dir> [<log-dir> [<vad-dir>]]
# Computes where there are moment of silence in the sessions
#   --vad-config           config passed to compute-vad-energy
sid/compute_vad_decision.sh --nj $nj --cmd "run.pl" ${data_in} exp/make_vad $vad_out
utils/fix_data_dir.sh ${data_in}

# prepare_feats_for_egs.sh <in-data-dir> <out-data-dir> <feat-dir>"
# Applies CMVN and removes nonspeech frames calculated by the vad
local/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd "run.pl" ${data_in} ${data_out} exp/$(basename ${data_out})
utils/fix_data_dir.sh ${data_out}
