typeset -i cnt
if [ $# -lt 1 ]
then
         #echo Usage: $0 machine.name
         #exit
         ip='192.168.20.10'
fi

echo $ip ...............
date

ping $ip 2>&1| ( while [ true ]
do
         read x
         #echo \"$x\"

         cnt=`echo $x | grep unknown | wc -l`
         if [ $cnt -ge 1 ]
         then
                 jdl $ip does not exist ...
                 exit
         fi

         cnt=`echo \"$x\" | grep ttl | wc -l`

         #echo cnt $cnt

         if [ $cnt -ge 1 ]
         then
                ./send.sh
                 exit
         else
                 sleep 1
         fi
done )
