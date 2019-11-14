import argparse
import numpy as np
from cryoem.projections import generate_2D_projections


parser = argparse.ArgumentParser(description="Generator of 2D projections of 3D Cryo-Em volumes")
parser.add_argument("--input","-mrc", help="Input file of 3D volume (*.mrc format)", 
                    type=str, required=True)
parser.add_argument("--proj-num", "-num", help="Number of 2D projections. Default 5000", 
                    type=int, default=5000)
parser.add_argument("--ang-coverage", "-cov", help="List of max values for each axis. E.g. `0.5,0.5,2.0` means it: x axis angle and y axis angle take values in range [0, 0.5*pi], z axis angles in range [0, 2.0*pi]", 
                    type=str, required=True)
parser.add_argument("--ang-shift", "-shift", help="Start of angular coverage", 
                    type=str, required=True)#, default="0.0,0.0,0.0" )
parser.add_argument("--output", "-mat", help="Name of output file containing projections with angles (with the extension)", 
                    default=None )
args = parser.parse_args()


AngCoverage = [float(x) for x in args.ang_coverage.split(',')]	
AngShift = [float(x) for x in args.ang_shift.split(',')]
print(AngShift)	
print(args.ang_shift)

generate_2D_projections(input_file_path=args.input, 
                        ProjNber=args.proj_num,
                        AngCoverage=AngCoverage,
                        AngShift=AngShift,
                        output_file_name=args.output)
