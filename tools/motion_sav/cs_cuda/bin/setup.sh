HN=`hostname`

case $HN in
medusa.*)
	export CUDA_CC=/usr/bin/gcc44
	export CUDA_CPLUSPLUS=/usr/bin/g++44
	;;
*)
	export CUDA_CC=/usr/bin/gcc
	export CUDA_CPLUSPLUS=/usr/bin/g++
	;;
esac
