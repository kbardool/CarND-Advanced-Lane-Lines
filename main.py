'''
Created on Apr 12, 2017

@author: kevin.bardool
'''
import sys
# print('classifydotbe : __name__ is ',__name__)
if __name__ == '__main__':
    REL_PATH = './'
    # sys.path.append(REL_PATH)
else:
    REL_PATH   = './'
    
    
import sys, os , io, time , argparse
from datetime                              import datetime
# from classes.mng_queue                     import MongoQueue
# from Main.URLLoader                        import URLLoader
# from Main.Crawler                          import CrawlUrls
# from Main.DataExtraction                   import DataExtraction
# from Main.Classification                   import Classify
    
INPUT_PATH            = REL_PATH 
OUTPUT_PATH           = REL_PATH 

''' 
#--------------------------------------------------------------------------
#-- MAIN 
#--------------------------------------------------------------------------'''
def main(ModelFiles, input_file, output_file, crawl_sites, classify_fs): 

    input_file  = INPUT_PATH  + input_file
    output_file = OUTPUT_PATH + output_file
 
    print('\n\n --> Classifydotbe started at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    

    file_sfx = datetime.now().strftime("%m%d%Y@%H%M")

    print('     Input File is   :', input_file)
    print('     Output File is  :', output_file)
    print()
    print()
    print('     Crawl websites  :  ', 'Yes' if crawl_sites == 'y' else 'No')
    print('     Output file     :  ', OUTPUT_PATH+args.output_file)

    """
    clear crawlqueue table from any previous runs
    """
    # queueTbl = MongoQueue()
    # queueTbl.clearTable()
   
    """
    Read CSV file of domains to classify/ load into crawling queue  
    """
    # urls  = URLLoader(input_file, skip_header=True, read_limit=350, write_limit=350)
 
    if len(urls) == 0:
        print('\n     !!!!! No URLs to process. Program will terminate....')
        return
    
    """
    Download websites 
    """
    # if crawl_sites == 'y':
        # if not CrawlUrls():
            # exit('\n      !!!!! An error occurred in the website crawling process ')

    
    """
    Perform feature extraction onx downloaded websites 
    """
    # FeatureFiles = DataExtraction(Urllist = urls)
    # if len(FeatureFiles) == 0:
        # print()
        # exit(' An error occurred in the data extraction process')

    """
    Perform classification and write output file
    """     
    # Classify(FeatureFiles, ModelFiles, classify_fs, output_file)
    
    
    print(' --> ALF Video ended at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    return
    
#--------------------------------------------------------------------------
#  Main Driver 
#--------------------------------------------------------------------------
if __name__ == '__main__':
    
    Models   = {     'HtmlText': ['LR_05_13_2017@1144'] ,
                     'MetaData': ['RF_05_12_2017@1731'],
                     'MetaText': ['LSVC_05_12_2017@1724']}
    """
    Verify input parms - Get input/output filenames from command line
    """
    parser = argparse.ArgumentParser(description='Load csv file of domains to classify ')
    parser.add_argument('input_file', metavar='Input_filename',  
                        help='Input file: Must be in .mp4 format ')
    parser.add_argument('output_file', metavar='Output_filename',  
                        help='Output destination ' )    
    # parser.add_argument('-fs', metavar='Feature set',  choices=['htmltext', 'metatext', 'metadata', 'all'], 
                        # default= 'htmltext', dest='classify_fs',
                        # help='Feature set to classify can be one of the following: htmltext, metatext, or metadata')    
    # parser.add_argument('-cs', metavar='Crawl sites',  choices=['y', 'n'], 
                        # default= 'y', dest='crawl_sites',
                        # help='Feature set to classify can be one of the following: htmltext, metatext, or metadata')                         
    args = parser.parse_args()
    start = time.time()
    print(' input file: ', input_file, '  output__file: ', output_file)
    # main(Models, args.input_file, args.output_file, args.crawl_sites, args.classify_fs)
    print()
    elapsed_time = time.time() - start
    print('     Elapsed time: %.2f seconds'%(elapsed_time))

    exit(' ALF Video completed normally')
