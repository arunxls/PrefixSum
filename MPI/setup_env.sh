#setup gcc
module load gcc/4.7.2
export OMPI_CC=/g/software/gcc/4.7.2/bin/gcc
export OMPI_CXX=/g/software/gcc/4.7.2/bin/g++

#setup openmpi
module load openmpi/1.4.3

#setup intel compilers
#this is sometimes needed for some libraries (libimf.so, libintlc.so.5)
#which are required by mpic++
module load intel/compilers/11.1.073

module load papi/4.2.0


