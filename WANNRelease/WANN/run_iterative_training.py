import subprocess
import numpy as np

for iteration in np.arange(2, 5):
    discfile = f'domain/mnist_discriminator_{iteration}.h5'
    wanfile  = f'log/mnistthirdrun{iteration}_cpu_best.out'
    discout  = f'domain/mnist_discriminator_{iteration + 1}.h5'
    
#    subprocess.call(['python3', 'update-discriminator.py', '-d', discfile, '-w', wanfile, '-o', discout, '-m', '1'])
    subprocess.call(['cp', discfile, 'domain/gan_discriminator.h5'])
    subprocess.call(['python3', 'wann_train.py', '-p', 'p/gan_mnist.json', '-o', f'mnistthirdrun{iteration+1}_cpu', '-n', '31', '-u', f'log/mnistthirdrun{iteration}_cpu_pop.obj'])
    
