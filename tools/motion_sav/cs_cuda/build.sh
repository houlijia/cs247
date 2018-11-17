# run this build file from its own directory

date
set -e
target=$1
for i in cs_misc cs_whm_encode
do
        cd $i
        make $target
        echo "$PWD  done ------------------------------"
        cd ..
done
cd ../RndStrmC
make $target
echo "$PWD  done ------------------------------"
cd -
cd proj
#for i in acq_ana acq_ana.bo acq_ana.file camera_demo
for i in acq_ana.file camera_demo obj_recog
    do
    echo "$PWD> proj/$i ............................................"
    cd $i
    if [ "$i" = "camera_demo" ]
    then
        cd tcp_socket
        make $target
	echo "$PWD  done ------------------------------"
        cd ..
    fi
    if [ "$i" = "obj_recog" ]
    then
        cd Nimresize
        make $target
	echo "$PWD  done ------------------------------"
        cd ..
    fi
    make $target
    echo "$PWD  done ------------------------------"
#ll yuvb420pcs
    cd ..
done

# cd ../../lfsr
# make $target
# echo "$PWD  done ------------------------------"
# ./gnrt_lfsr -w. 20
# cd ../cs_cuda

echo "----------------------------------------------------"

