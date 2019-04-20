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

mpirun -np 2 $RUNNER -m WDmodel --specfile WDmodel/tests/test.flm --photfile WDmodel/tests/test.phot --rebin 2 --everyn 3 --mpi --redo --trimspec 3700 5200 --samptype pt --ntemps 2 --nburnin 20 --nprod 100 --nwalkers 100 --covtype SHO --tau_fix True --tau 5000 --lamshift 2 --vel -50 --reddeningmodel f99 --excludepb F160W --rescale --blotch --phot_dispersion 0.001 --thin 2 --skipminuit --dl 380 --trimspec 3700 5200
mpirun -np 2 $RUNNER -m WDmodel --specfile WDmodel/tests/test.flm --resume --nprod 5 --mpil
mpirun -np 2 $RUNNER -m WDmodel --specfile WDmodel/tests/test.flm --ignorephot --rebin 2 --everyn 3 --nburnin 10 --nprod 50 --mpi --redo --nwalkers 100 --reddeningmodel od94
if [ "$1" = -c ]; then
    coverage combine
fi
$ARUNNER -m WDmodel --specfile WDmodel/tests/test.flm --ignorephot --nburnin 10 --nprod 50 --redo --outdir out_temp --covtype Exp --nwalkers 50 --av_fix True --av 0.03 --savefig --mu None --samptype gibbs --reddeningmodel ccm89
$ARUNNER -m WDmodel --specfile WDmodel/tests/test.flm --ignorephot --nprod 50 --nburnin 10 --redo --nwalkers 30 --covtype White --reddeningmodel custom
$ARUNNER test_WDmodel.py

if [ "$1" = -c ]; then
    echo "travis_fold:end:FIT Fitting test data done"
fi
