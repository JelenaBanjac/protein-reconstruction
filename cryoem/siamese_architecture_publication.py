"""
Code used to genrate the neural network architecture form our paper.
Relies on the code developed at the following link: https://github.com/HarisIqbal88/PlotNeuralNet
"""

import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
# arch = [
#     to_head( '..' ),
#     to_cor(),
#     to_begin(),
#     to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2 ),
#     to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
#     to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
#     to_connection( "pool1", "conv2"),
#     to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
#     to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
#     to_connection("pool2", "soft1"),
#     to_end()
#     ]

scale = 0.5
scale2 = 0.5

arch = [
    
    to_head( '..' ),
    to_cor(),
    to_begin(),
    
    #### LOWER ####
    to_input( '5j0n_noise16.png', to='(-5,0,0)', width=27, height=27),
    #to_Conv("conv1", 116, 3, offset="(0,0,0)", to="(0,0,0)", height=116, depth=116, width=3 ),
    #to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
    to_Conv("conv1", 116, 32, offset="(1,0,0)", to="(image-east)", height=116, depth=116, width=32*scale ),

    to_Pool("pool1", offset="(-3,0,0)", to="(conv1-east)", height=58, depth=58, width=32*scale),
    to_Conv("conv2", 58, 64, offset="(5,0,0)", to="(pool1-east)", height=58, depth=58, width=64*scale ),
    to_connection( "pool1", "conv2"),

    to_Pool("pool2", offset="(-6.5,0,0)", to="(conv2-east)", height=29, depth=29, width=64*scale),
    to_Conv("conv3", 29, 128, offset="(5,0,0)", to="(pool2-east)", height=29, depth=29, width=128*scale ),
    to_connection( "pool2", "conv3"),

    to_Pool("pool3", offset="(-13,0,0)", to="(conv3-east)", height=15, depth=15, width=128*scale),
    to_Conv("conv4", 15, 256, offset="(5,0,0)", to="(pool3-east)", height=15, depth=15, width=256*scale2 ),
    to_connection( "pool3", "conv4"),

    to_Pool("pool4", offset="(-25.5,0,0)", to="(conv4-east)", height=8, depth=8, width=256*scale2),
    to_Conv("conv5", 8, 256, offset="(5,0,0)", to="(pool4-east)", height=8, depth=8, width=256*scale2 ),
    to_connection( "pool4", "conv5"),

    to_Pool("pool5", offset="(-25.5,0,0)", to="(conv5-east)", height=4, depth=4, width=256*scale2),
    to_Conv("conv6", 4, 512, offset="(5,0,0)", to="(pool5-east)", height=4, depth=4, width=512*scale2 ),
    to_connection( "pool5", "conv6"),

    to_Pool("pool6", offset="(-51,0,0)", to="(conv6-east)", height=2, depth=2, width=512*scale2),
    to_Conv("conv7", 2, 512, offset="(5,0,0)", to="(pool6-east)", height=2, depth=2, width=512*scale2 ),
    to_connection( "pool6", "conv7"),

    to_Pool("pool7", offset="(-51.2,0,0)", to="(conv7-east)", height=1, depth=1, width=512*scale2),
    to_UnPool("flatten1", offset="(3,0,0)", to="(conv7-east)", width=1, height=1, depth=512*scale2, opacity=0.5, caption=" "),
    to_connection( "pool7", "flatten1"),

    #### UPPER ####
    to_input( '5j0n_noise16.png', to='(-5,36,0)', width=27, height=27),
    #to_Conv("conv1", 116, 3, offset="(0,0,0)", to="(0,0,0)", height=116, depth=116, width=3 ),
    #to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
    to_Conv("conv1", 116, 32, offset="(1,36,0)", to="(image-east)", height=116, depth=116, width=32*scale ),

    to_Pool("pool1", offset="(-3,0,0)", to="(conv1-east)", height=58, depth=58, width=32*scale),
    to_Conv("conv2", 58, 64, offset="(5,0,0)", to="(pool1-east)", height=58, depth=58, width=64*scale ),
    to_connection( "pool1", "conv2"),

    to_Pool("pool2", offset="(-6.5,0,0)", to="(conv2-east)", height=29, depth=29, width=64*scale),
    to_Conv("conv3", 29, 128, offset="(5,0,0)", to="(pool2-east)", height=29, depth=29, width=128*scale ),
    to_connection( "pool2", "conv3"),

    to_Pool("pool3", offset="(-13,0,0)", to="(conv3-east)", height=15, depth=15, width=128*scale),
    to_Conv("conv4", 15, 256, offset="(5,0,0)", to="(pool3-east)", height=15, depth=15, width=256*scale2 ),
    to_connection( "pool3", "conv4"),

    to_Pool("pool4", offset="(-25.5,0,0)", to="(conv4-east)", height=8, depth=8, width=256*scale2),
    to_Conv("conv5", 8, 256, offset="(5,0,0)", to="(pool4-east)", height=8, depth=8, width=256*scale2 ),
    to_connection( "pool4", "conv5"),

    to_Pool("pool5", offset="(-25.5,0,0)", to="(conv5-east)", height=4, depth=4, width=256*scale2),
    to_Conv("conv6", 4, 512, offset="(5,0,0)", to="(pool5-east)", height=4, depth=4, width=512*scale2 ),
    to_connection( "pool5", "conv6"),

    to_Pool("pool6", offset="(-51,0,0)", to="(conv6-east)", height=2, depth=2, width=512*scale2),
    to_Conv("conv7", 2, 512, offset="(5,0,0)", to="(pool6-east)", height=2, depth=2, width=512*scale2 ),
    to_connection( "pool6", "conv7"),

    to_Pool("pool7", offset="(-51.2,0,0)", to="(conv7-east)", height=1, depth=1, width=512*scale2),
    to_UnPool("flatten2", offset="(3,0,0)", to="(conv7-east)", width=1, height=1, depth=512*scale2, opacity=0.5, caption=" "),
    to_connection( "pool7", "flatten2"),

    #### connect to lambda
    to_UnPool("concatenate", offset="(20,24,0)", to="(flatten1-east)", width=2, height=1, depth=512*scale2, opacity=0.5, caption=" "),
    to_connection( "flatten1", "concatenate"),
    #to_UnPool("lambda", offset="(30,-15,0)", to="(flatten2-east)", width=1, height=1, depth=1, opacity=0.5, caption=" "),
    to_connection( "flatten2", "concatenate"),
    to_UnPool("lambda", offset="(30,24,0)", to="(flatten1-east)", width=1, height=1, depth=1, opacity=0.5, caption=" "),
    to_connection( "concatenate", "lambda"),

    #to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
    #to_connection("pool2", "soft1"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
