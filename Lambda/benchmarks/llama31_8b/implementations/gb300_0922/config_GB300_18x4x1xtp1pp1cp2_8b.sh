source $(dirname ${BASH_SOURCE[0]})/config_GB200_18x4x1xtp1pp1cp2_8b.sh

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
