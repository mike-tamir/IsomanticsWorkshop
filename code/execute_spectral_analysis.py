import sys
import json
from ismtools import parse_arguments, extract_T_matrix_dict, T_matrix_point_stats


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Please specify the relative path to the 'T_matrices/' dir containing translation csv files.")
        print("EXAMPLE: ../data/l2=0_01_normality=0_000001/")

    else:
        ### Specify paths ###
        # append "T_matrices to the inpath
        T_matrix_dir = sys.argv[1]+"T_matrices"

        # specify outpath for json
        out_json_path = sys.argv[1]+"spec_analysis_stats.json"

        # execute statistical analysis of translations spectra
        full = extract_T_matrix_dict(T_matrix_dir=T_matrix_dir,
                                     stats=['min','max','mean','median','std'],
                                     calc_cov=True,
                                     calc_inv=False,
                                     calc_SVD=True,
                                     log_spectrum=True,
                                     normality=True,
                                     determinant=True,
                                     verbose=0
                                    )

        print("Translation Spectra Analyzed writing json to the following path:\n%s" % (out_json_path))

        # Write dict out as json
        out_dict = T_matrix_point_stats(full)

        with open(out_json_path,"w") as jpath:
            json.dump(out_dict, jpath)

        jpath.close()
