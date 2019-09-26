import argparse
import numpy as np
from cryoem.projections import generate_2D_projections


parser = argparse.ArgumentParser(description="Generator of 2D projections of 3D Cryo-Em volumes")
parser.add_argument("--input","-mrc", help="Input file of 3D volume (*.mrc format)", 
                    type=str, required=True)
parser.add_argument("--proj-num", "-num", help="Number of 2D projections. Default 50000", 
                    type=int, default=50000)
parser.add_argument("--ang-coverage", "-cov", help="Angular coverage (0.5: half-sphere, 1: complete sphere). Default 0.5", 
                    type=float, default=0.5)
parser.add_argument("--ang-shift", "-shift", help="Start of angular coverage. Default pi/2", 
                    type=float, default=np.pi/2 )
parser.add_argument("--output", "-mat", help="Name of output file containing projections with angles (with the extension)", 
                    default=None )
args = parser.parse_args()

generate_2D_projections(input_file_path=args.input, 
                        ProjNber=args.proj_num,
                        AngCoverage=args.ang_coverage,
                        AngShift=args.ang_shift,
                        output_file_name=args.output)
