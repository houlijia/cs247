#-n 18631281 
iperf3 -u -c 192.168.20.10 -p 8042 -F /home/jianwel/all_cif_300.csvid -t 15 -i 1 -f k -b 275K | tee iperf.log
