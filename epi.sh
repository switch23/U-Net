#!/bin/sh

#$ -cwd
#$ -V -S /bin/bash
#$ -N epi
#$ -q [使用するGPUサーバーを指定]

rm -rf input
rm -rf label
rm -rf predicted

mkdir input
mkdir label
mkdir predicted

# >>> conda init >>>
__conda_setup="$(CONDA_REPORT_ERRORS=false '$HOME/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="$PATH:$HOME/anaconda3/bin"
    fi
fi
unset __conda_setup
# <<< conda init <<<

conda activate py37
CUDA_VISIBLE_DEVICES=0 python epi.py
