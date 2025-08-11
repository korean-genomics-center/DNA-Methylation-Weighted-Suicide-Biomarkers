import os
import argparse


def run(input_file, output_file):

    with open (input_file) as fhin:
        with open(output_file,'w') as fhout:
            headerline=fhin.readline()
            fhout.write(headerline)

            prevstrand='-'
            curline=fhin.readline()
            while curline :
                curchr,curstart,curstrand,curcount,curfreqc = parse_a_methyl_cpg_row(curline)
                if(curstrand=='+'): 
                    #case curstrand '+'
                    #Do nothing and handle info for next line
                    prevchr,prevstart,prevstrand,prevcount,prevfreqc = curchr,curstart,curstrand,curcount,curfreqc
                else: 
                    #case curstrand=='-'
                    if(prevstrand=='+'): 
                        #case curstrand == '-'
                        # and prevstrand =='+'
                        if(prevchr==curchr and prevstart==curstart-1):
                            # if prev and curstrand paired,
                            newcount=prevcount+curcount
                            newC=prevcount*prevfreqc+curfreqc*curcount
                            newfreqC=newC/newcount
                            newout=f"{prevchr}\t{prevstart}\t{prevstrand}\t{newcount}\t{newfreqC}\n"
                            fhout.write(newout)
                        else:
                            fhout.write(prevline)
                            #case curstrand == '-'
                            # and prevstrand =='+'
                            #but not paired
                            newout=f"{curchr}\t{curstart-1}\t+\t{curcount}\t{curfreqc}\n"
                            fhout.write(newout)

                    else: #case prev strand =='-'
                        newout=f"{curchr}\t{curstart-1}\t+\t{curcount}\t{curfreqc}\n"
                        fhout.write(newout)
                prevline=curline
                prevchr,prevstart,prevstrand,prevcount,prevfreqc = curchr,curstart,curstrand,curcount,curfreqc
                curline=fhin.readline()

def parse_a_methyl_cpg_row(aline):
    achr,astart,astrand,acount,afreqc=aline.split("\t")
    return achr, int(astart), astrand, int(acount), float(afreqc)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input", help = "Path to CpG Value Plain Text")
    argument_parser.add_argument("--output", help = "Path to save Strand-Merged Plain Text")
    
    args = argument_parser.parse_args()
    
    run(
        args.input,
        args.output
    )