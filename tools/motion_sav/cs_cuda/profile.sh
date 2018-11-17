export COMPUTE_PROFILE=1
export COMPUTE_PROFILE_LOG=/tmp/p.log
export COMPUTE_PROFILE_CONFIG=/home/ldl/mr/cuda/cs/profile.option

echo $# ... 

if [ $# -eq 0 ]
then
	export COMPUTE_PROFILE_CSV=0
	echo not use CSV ...
else
	export COMPUTE_PROFILE_CSV=1
	echo use CSV ...
fi
