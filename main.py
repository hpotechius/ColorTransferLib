"""
Copyright 2025 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from test import *
import argparse


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# args.test:
#   - "all_CT": Test all color transfer methods for image-to-image transfer
#   - "all_ST": Test one color transfer methods for all possible data type combinations
#   - "all_CZ": Test all style transfer methods for image-to-image transfer
#   - "all_EVAL" Test evaluation metrics on src, ref and out images
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':  

    parser = argparse.ArgumentParser(description='Color Transfer Script')
    parser.add_argument('--test', type=str, required=False, help='Type of test to run')
    parser.add_argument('--out_path', type=str, required=False, help='Reference path')
    args = parser.parse_args()

    if args.test == "all_CT":
        test_all_CT_all_datatypes(args.out_path)
    elif args.test == "all_ST":
        test_all_ST_all_datatypes(args.out_path)
    elif args.test == "all_CZ":
        test_all_CZ_all_datatypes(args.out_path)
    elif args.test == "all_EVAL":
        test_all_EVAL()


    # Example without the ColorTransfer Class
    
    # from ColorTransferLib.Algorithms.FUZ.FUZ import FUZ
    # from ColorTransferLib.Utils.BaseOptions import BaseOptions
    # import json
    # with open("ColorTransferLib/Options/FUZ.json", 'r') as f:
    #     options = json.load(f)
    #     opt = BaseOptions(options)
    # #src = Image(file_path='testdata/images/256_interior-02.png')
    # src = Image(file_path='testdata/images/The_Kiss.png')
    # #src = Video(file_path='testdata/videos/output.mp4')
    # ref = Image(file_path='testdata/images/The_Scream.png')
    # out = FUZ.apply(src, ref, opt)    
    # if out["status_code"] == 0:
    #     out["object"].write("testdata/results/out_FUZ")
    # else:
    #     print("Error: " + out["response"])
