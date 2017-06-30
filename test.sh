#! /bin/bash
set -ev
if [ "$1" = -c ]; then
    RUNNER="coverage run -p --source=WDmodel"
    ARUNNER="coverage run -a --source=WDmodel"
    echo "travis_fold:start:FIT Fitting test data"
else
    RUNNER="python"
    ARUNNER="python"
fi

mpirun -np 4 $RUNNER -m WDmodel --specfile WDmodel/tests/test.flm --ignorephot --rebin 2 --everyn 3 --nprod 100 --mpi --redo 
if [ "$1" = -c ]; then
    coverage combine
fi
$ARUNNER -m WDmodel --specfile WDmodel/tests/test.flm --ignorephot --nprod 100 --resume

if [ "$1" = -c ]; then
    echo "travis_fold:end:FIT Fitting test data done"
fi
