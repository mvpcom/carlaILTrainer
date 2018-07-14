# Programmed by Mojtaba Valipour @ Shiraz University - 2018 - vpcom.ir
# Based on the CVC code available in the github repository
# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

## Auto table generator for the paper writing
## Tables are based on the CoRL 2017 Carla Team Paper
## This help me to have a very beautiful tables easier
## Output Support: Text, Latex, HTML

# System Test: Ubuntu 16.04 LTS
# Python: 2.7.14, conda 4.3.30
# Environment: perfectEnv.yaml = carlaSimPy2

# Example: python calc_CORL2017_Tables.py --path "./_benchmarks_results/test/" -v -n "CoRL-2017 Carla Paper"

### "./_benchmarks_results/test/" contains the following folders: 
### CarlaPaperModel_Test01_CoRL2017_Town01
### CarlaPaperModel_Test01_CoRL2017_Town02
### CarlaPaperModel_Test02_CoRL2017_Town01
### CarlaPaperModel_Test02_CoRL2017_Town02
### CarlaPaperModel_Test03_CoRL2017_Town01
### CarlaPaperModel_Test03_CoRL2017_Town02
##### Each one includes measurements.csv, summary.csv and log_ files

import abc
import argparse
import math
import time
import numpy as np
import logging
import glob
from tabulate import tabulate # only for the presentation

from carla.driving_benchmark.experiment_suites import CoRL2017
from carla.driving_benchmark.metrics import Metrics
from carla.driving_benchmark import results_printer

# Save tables as html file
htmlWrapper = """
<html>
<head>
<style>
table{
    font-family: arial, sans-serif;
    border-collapse: collapse;
    width: 100%%;
}
td, th {
    border: 1px solid #dddddd;
    text-align: left;
    padding: 8px;
}
tr:nth-child(even){
    background-color: #dddddd;
}
</style>
<title> Self-Driving Car Research </title>
<p>By Mojtaba Valipour @ Shiraz University - 2018 </p>
<p><a href="http://vpcom.ir/">vpcom.ir</a></p>
</head>
<body><p>MODEL: %s</a></p><p>%s</p><p>%s</p><p>%s</p><p>%s</p><p>%s</p><p>%s</p><p>%s</p><p>%s</p><p>%s</p></body>
</html>
"""

# Tested by latexbase.com
latexWrapper = """
\\documentclass{article}
\\usepackage{graphicx}

\\begin{document}

\\title{Self-Driving Car Research}
\\author{Mojtaba Valipour}

\\maketitle

\\section{Model : %s}

\\subsection{Tables}
\\subsubsection{Percentage of Success}
\\begin{center}
%s
Success rate for the agent (mean and standard deviation shown).
\\end{center}
\\subsubsection{Infractions : Straight}
\\begin{center}
%s
Average number of kilometers travelled before an infraction. 
\\end{center}
\\subsubsection{Infractions : One Turn}
\\begin{center}
%s
Average number of kilometers travelled before an infraction. 
\\end{center}
\\subsubsection{Infractions : Navigation}
\\begin{center}
%s
Average number of kilometers travelled before an infraction. 
\\end{center}
\\subsubsection{Infractions : Navigation With Dynamic Obstacles}
\\begin{center}
%s
Average number of kilometers travelled before an infraction. 
\\end{center}

\\subsubsection{Num Infractions : Straight}
\\begin{center}
%s
Number of infractions occured in the whole path
\\end{center}
\\subsubsection{Num Infractions : One Turn}
\\begin{center}
%s
Number of infractions occured in the whole path
\\end{center}
\\subsubsection{Num Infractions : Navigation}
\\begin{center}
%s
Number of infractions occured in the whole path
\\end{center}
\\subsection{Num Infractions : Navigation With Dynamic Obstacles}
\\begin{center}
%s
Number of infractions occured in the whole path
\\end{center}

\\end{document}
"""

if (__name__ == '__main__'):
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '-n', '--model_name',
        metavar='T',
        default='CoRL2017-Paper',
        help='The name of the model for writing in the reports'
    )
    argparser.add_argument(
        '-p', '--path',
        metavar='P',
        default='test',
        help='Path to all log files'
    )

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('sarting the calculations %s', "0") #TODO: add time instead on zero

    experiment_suite = CoRL2017("Town01")

    metrics_object = Metrics(experiment_suite.metrics_parameters,
                                 experiment_suite.dynamic_tasks)

    # Improve readability by adding a weather dictionary
    weather_name_dict = {1: 'Clear Noon', 3: 'After Rain Noon',
                         6: 'Heavy Rain Noon', 8: 'Clear Sunset',
                         4: 'Cloudy After Rain', 14: 'Soft Rain Sunset'}
    
    
    # names for all the test logs
    pathNames = {0:'_Test01_CoRL2017_Town01',
                 1:'_Test02_CoRL2017_Town01',
                 2:'_Test03_CoRL2017_Town01',
                 3:'_Test01_CoRL2017_Town02',
                 4:'_Test02_CoRL2017_Town02',
                 5:'_Test03_CoRL2017_Town02'}

    tasksSuccessRate = {0: 'Straight', 1: 'One Turn', 2: 'Navigation', 3: 'Nav. Dynamic'} # number_of_episodes = len(list(metrics_summary['episodes_fully_completed'].items())[0][1])
    tasksInfractions = {0: 'Opposite Lane', 1: 'Sidewalk', 2: 'Collision-static', 3: 'Collision-car', 4:'Collision-pedestrian'} # 
    states = {0: 'Training Conditions', 1: 'New Town', 2: 'New Weather', 3: 'New Town & Weather'}
    statesSettings = {0: {'Path':[pathNames[0],pathNames[1],pathNames[2]], 'Weathers':experiment_suite.train_weathers},
                      1: {'Path':[pathNames[3],pathNames[4],pathNames[5]], 'Weathers':experiment_suite.train_weathers},
                      2: {'Path':[pathNames[0],pathNames[1],pathNames[2]], 'Weathers':experiment_suite.test_weathers},
                      3: {'Path':[pathNames[3],pathNames[4],pathNames[5]], 'Weathers':experiment_suite.train_weathers+experiment_suite.test_weathers}}
    
    # In CoRL-2017 paper, infraction are only computed on the fourth task - "Navigation with dynamic obstacles".
    dataSuccessRate = np.zeros((len(tasksSuccessRate),len(states))) # hold the whole table 1 data
    dataInfractions = np.zeros((len(tasksSuccessRate),len(tasksInfractions),len(states))) # hold the whole table 2 data
    dataNumInfractions = np.zeros((len(tasksSuccessRate),len(tasksInfractions),len(states))) # hold the whole table 3 data
    dataSuccessRateSTD = np.zeros((len(tasksSuccessRate),len(states))) # hold the whole table 1 std data
    dataInfractionsSTD = np.zeros((len(tasksSuccessRate),len(tasksInfractions),len(states))) # hold the whole table 2 std data
    dataNumInfractionsSTD = np.zeros((len(tasksSuccessRate),len(tasksInfractions),len(states))) # hold the whole table 3 std data

    # TABLE 1 - CoRL2017 Paper
    metrics_to_average = [
        'episodes_fully_completed',
        'episodes_completion'
    ]

    infraction_metrics = [
        'collision_pedestrians',
        'collision_vehicles',
        'collision_other',
        'intersection_offroad',
        'intersection_otherlane'

    ]

    # Configuration
    table1Flag = True
    table2Flag = True
    table3Flag = True

    # extract the start name of the folders
    #TODO: Automatic this extraction process better and smartly
    if args.path[-1]=='/':
	addSlashFlag = False
	allDir = glob.glob(args.path+'*')
    else:
	addSlashFlag = True
	allDir = glob.glob(args.path+'/*') 
    extractedPath = allDir[0].split('/')[-1].replace(statesSettings[0]['Path'][0],'')
    logging.info('Please make sure all the subdirectory of %s start with %s', args.path, extractedPath)

    for sIdx, state in enumerate(states):
        logging.debug('State: %s', state)
        weathers = statesSettings[state]['Weathers']
        allPath = statesSettings[state]['Path']

        # This will make life easier for calculating std 
        dataListTable1 = [[] for i in range(len(tasksSuccessRate))] 
        dataListTable2 = [[[] for i in range(len(tasksSuccessRate))] for i in range(len(tasksInfractions))] 
	dataListTable3 = [[[] for i in range(len(tasksSuccessRate))] for i in range(len(tasksInfractions))] 
        #logging.debug("Data list table 2 init: %s",dataListTable2)

        # calculate metrics : episodes_fully_completed	
        for p in allPath: 
	    if addSlashFlag == True:
	            path = args.path + '/' + extractedPath + p
	    else:
	            path = args.path + extractedPath + p
            metrics_summary = metrics_object.compute(path)
            number_of_tasks = len(list(metrics_summary[metrics_to_average[0]].items())[0][1])
            values = metrics_summary[metrics_to_average[0]] # episodes_fully_completed
            
            if(table1Flag):
                logging.debug("Working on table 1 ...")
                metric_sum_values = np.zeros(number_of_tasks)
                for w, tasks in values.items():
                    if w in set(weathers):
                        count = 0
                        for tIdx, t in enumerate(tasks):
                            #print(weathers[tIdx]) #float(sum(t)) / float(len(t)))
                            metric_sum_values[count] += (float(sum(t)) / float(len(t))) * 1.0 / float(len(weathers))
                            count += 1
                
                # array's elements displacement, this is because of std/avg calculation
                for j in range(number_of_tasks):
                        dataListTable1[j].append(metric_sum_values[j])
            
            # table 2
            if(table2Flag):
                logging.debug("Working on table 2 and 3 ...")
                for metricIdx, metric in enumerate(infraction_metrics):
                    values_driven = metrics_summary['driven_kilometers']
                    values = metrics_summary[metric]
                    metric_sum_values = np.zeros(number_of_tasks)
                    summed_driven_kilometers = np.zeros(number_of_tasks)

                    for items_metric, items_driven in zip(values.items(), values_driven.items()):
                        w = items_metric[0] # weather
                        tasks = items_metric[1]
                        tasks_driven = items_driven[1]

                        if w in set(weathers):
                            count = 0
                            for t, t_driven in zip(tasks, tasks_driven):
                                #logging.debug("t_driven: %s \n t: %s \n tSum: %f", t_driven, t, float(sum(t)))
                                metric_sum_values[count] += float(sum(t))
                                summed_driven_kilometers[count] += t_driven

                                count += 1
                    # array's elements displacement, this is because of std/avg calculation
                    for i in range(number_of_tasks):
			dataListTable3[metricIdx][i].append(metric_sum_values[i])
                        if metric_sum_values[i] == 0:
                            dataListTable2[metricIdx][i].append(summed_driven_kilometers[i])
                        else:
                            dataListTable2[metricIdx][i].append(summed_driven_kilometers[i] / metric_sum_values[i])
                        
                    #print(dataListTable2)

        if(table1Flag):
            # Accumulate the whole results and calculate std and avg
            for tIdx, t in enumerate(dataListTable1):
                dataSuccessRate[tIdx][sIdx] = np.mean(t)
                dataSuccessRateSTD[tIdx][sIdx] = np.std(t)
        
        
            #print(dataSuccessRate[tIdx][sIdx], ' +/- ', dataSuccessRateSTD[tIdx][sIdx])

        if(table2Flag):
            for metricIdx in range(len(infraction_metrics)):
                tmp = dataListTable2[metricIdx]
                for tIdx,t in enumerate(tmp):
                    # Accumulate the whole results and calculate std and avg
                    # fill in reverse because infraction matrics is reverse considering the table output
                    dataInfractions[tIdx][len(infraction_metrics)-1-metricIdx][sIdx] = np.mean(t)
                    dataInfractionsSTD[tIdx][len(infraction_metrics)-1-metricIdx][sIdx] = np.std(t)
        
        if(table3Flag):
            for metricIdx in range(len(infraction_metrics)):
                tmp = dataListTable3[metricIdx]
                for tIdx,t in enumerate(tmp):
                    # Accumulate the whole results and calculate std and avg
                    # fill in reverse because infraction matrics is reverse considering the table output
                    dataNumInfractions[tIdx][len(infraction_metrics)-1-metricIdx][sIdx] = np.mean(t)
                    dataNumInfractionsSTD[tIdx][len(infraction_metrics)-1-metricIdx][sIdx] = np.std(t)
        

    # Open external files
    fHtml = open(args.path+'/results.html','w')

    fLaTex = open(args.path+'/results.laTex','w') # TODO: Fix this later 
    # This is not an actual laTex format, you should copy this results into a real one ;P 
    
    # This is only for a good presentation not for calculation
    tableSRRows = []
    tableSRHeaders = ['Tasks']
    tableSRHeaders.extend(states.values())

    allTablesListHtml = []
    allTablesListLaTex = []
    if(table1Flag):
        for tIdx, t in enumerate(dataListTable1): # for each tasks
            row = [tasksSuccessRate[tIdx]]
            for sIdx, state in enumerate(states): # for each states
                row.append("".join([str(round(dataSuccessRate[tIdx][sIdx],2)), ' +/- ', str(round(dataSuccessRateSTD[tIdx][sIdx],2))])) 
            tableSRRows.append(row)
        
        print("\nPercentage of Success")
        print(tabulate(tableSRRows, headers=tableSRHeaders, tablefmt = 'orgtbl')) 
        allTablesListHtml.append(tabulate(tableSRRows, headers=tableSRHeaders, tablefmt = 'html'))
        allTablesListLaTex.append(tabulate(tableSRRows, headers=tableSRHeaders, tablefmt = 'latex'))
    # This is only for a good presentation not for calculation
    if(table2Flag):
        for taskIdx in tasksSuccessRate: # for each tasks
            tableSRRows = []
            tableSRHeaders = ['Infractions']
            tableSRHeaders.extend(states.values())
            print("\n Task: %s \n" % tasksSuccessRate[taskIdx])
            for metricIdx, metric in enumerate(infraction_metrics): 
                #print(metricIdx, metric)
                row = [tasksInfractions[metricIdx]]
                for sIdx, state in enumerate(states): # for each states
                    row.append("".join([str(round(dataInfractions[taskIdx][metricIdx][sIdx],2)), ' +/- ', str(round(dataInfractionsSTD[taskIdx][metricIdx][sIdx],2))])) 
                tableSRRows.append(row)
            print(tabulate(tableSRRows, headers=tableSRHeaders, tablefmt = 'orgtbl')) 
            allTablesListHtml.append(tabulate(tableSRRows, headers=tableSRHeaders, tablefmt = 'html'))
            allTablesListLaTex.append(tabulate(tableSRRows, headers=tableSRHeaders, tablefmt = 'latex'))

    if(table3Flag):
        for taskIdx in tasksSuccessRate: # for each tasks
            tableSRRows = []
            tableSRHeaders = ['Number of Infractions']
            tableSRHeaders.extend(states.values())
            print("\n Task: %s \n" % tasksSuccessRate[taskIdx])
            for metricIdx, metric in enumerate(infraction_metrics): 
                #print(metricIdx, metric)
                row = [tasksInfractions[metricIdx]]
                for sIdx, state in enumerate(states): # for each states
                    row.append("".join([str(round(dataNumInfractions[taskIdx][metricIdx][sIdx],2)), ' +/- ', str(round(dataNumInfractionsSTD[taskIdx][metricIdx][sIdx],2))])) 
                tableSRRows.append(row)
            print(tabulate(tableSRRows, headers=tableSRHeaders, tablefmt = 'orgtbl')) 
            allTablesListHtml.append(tabulate(tableSRRows, headers=tableSRHeaders, tablefmt = 'html'))
            allTablesListLaTex.append(tabulate(tableSRRows, headers=tableSRHeaders, tablefmt = 'latex'))

    # stream into the files
    htmlBody = htmlWrapper % (args.model_name, allTablesListHtml[0], allTablesListHtml[1], allTablesListHtml[2], allTablesListHtml[3]
                         ,allTablesListHtml[4], allTablesListHtml[5], allTablesListHtml[6], allTablesListHtml[7]
                         ,allTablesListHtml[8]) # TODO: Check if unpacking using * works
    latexBody = latexWrapper % (args.model_name, allTablesListLaTex[0], allTablesListLaTex[1], allTablesListLaTex[2], allTablesListLaTex[3]
                         ,allTablesListLaTex[4], allTablesListLaTex[5], allTablesListLaTex[6], allTablesListLaTex[7]
                         ,allTablesListLaTex[8]) # TODO: Check if unpacking using * works
    
    fHtml.write(htmlBody)
    fHtml.close()
    fLaTex.write(latexBody.replace('+/-','${\pm}$'))
    fLaTex.close()

    #metrics_summary = metrics_object.compute(args.path)

                        
    # # print details
    # print("")
    # print("")
    # print("----- Printing results for training weathers (Seen in Training) -----")
    # print("")
    # print("")
    # results_printer.print_summary(metrics_summary, experiment_suite.train_weathers,
    #                                 args.path)

    # print("")
    # print("")
    # print("----- Printing results for test weathers (Unseen in Training) -----")
    # print("")
    # print("")

    # results_printer.print_summary(metrics_summary, experiment_suite.test_weathers,
    #                                 args.path)   

    
