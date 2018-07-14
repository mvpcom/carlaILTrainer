# carlaILTrainer
Carla Imitation Learning Trainer


The main project belongs to Carla Team, and I just wrote a training system for the main project. [Here](https://github.com/carla-simulator/imitation-learning) you can find more information about the original project.


Please check out [issue](https://github.com/carla-simulator/imitation-learning/issues/1) and [issue](https://github.com/carla-simulator/carla/issues/580) to find out why this code was published. 


TODO:
* Run all branches , but just back propagate on one ( use a mask for that).
* Use the Tensorflow queues for data loading
* Compatibility fix for carla checkpoint loading 
* Fix the loss function (Use only one loss)
* Balance the dataset according to labels in training mode
* Test the whole system with the recent benchmark

## Contribution:
Please feel free to pull a request. Any features and any changes are welcome. 

All Contributors: 
* Mojtaba Va'lipour: [@mvpcom](https://github.com/mvpcom)
* Ivan Chernuha : [@merryHunter](https://github.com/merryHunter)
## Paper Results:

This is an auto table generator for the paper writing purpose. Tables are based on the CoRL 2017 Carla Team Paper. This can help me to have a gorgeous table easier and with a lot of fewer efforts. Currently, Text, LaTex, and HTML are supported. Before you are going to run this code, you have to prepare the results for three different runs for each town in Carla. You can see an example in the following passage.

### Output Support: Text, Latex, HTML

### Example: python calc_CORL2017_Tables.py --path "./CarlaPaper_ReExperiment/" -v -n "CoRL-2017 Carla Paper"

### "./CarlaPaper_ReExperiment/" contains the following folders: 
```
CarlaPaper_ReExperiment/
├── CarlaPaperModel_Test01_CoRL2017_Town01
│   ├── log_201807010542
│   ├── log_201807011830
│   ├── log_201807020224
│   ├── measurements.csv
│   ├── metrics.json
│   ├── res
│   └── summary.csv
├── CarlaPaperModel_Test01_CoRL2017_Town02
│   ├── log_201807031155
│   ├── log_201807032256
│   ├── log_201807050552
│   ├── measurements.csv
│   └── summary.csv
├── CarlaPaperModel_Test02_CoRL2017_Town01
│   ├── log_201807020250
│   ├── log_201807020739
│   ├── log_201807021521
│   ├── measurements.csv
│   └── summary.csv
├── CarlaPaperModel_Test02_CoRL2017_Town02
│   ├── log_201807050555
│   ├── measurements.csv
│   └── summary.csv
├── CarlaPaperModel_Test03_CoRL2017_Town01
│   ├── log_201807022050
│   ├── log_201807030243
│   ├── log_201807030738
│   ├── measurements.csv
│   └── summary.csv
├── CarlaPaperModel_Test03_CoRL2017_Town02
│   ├── log_201807052015
│   ├── log_201807060744
│   ├── measurements.csv
│   └── summary.csv
├── results.html
└── results.laTex
```

P.S: results.html and results.laTex are produced by this code after execution. These are a little bit different because of different configuration and different Carla version. In these results, bikes were in the test only. Please share your code and enhancements by a pull request with the world if you use this in your papers and add a feature.

### Example Output:

<html>
<body><p>MODEL: CoRL-2017 Carla Paper_ReExperiment</a></p><p><table>
<thead>
<tr><th>Tasks       </th><th>Training Conditions  </th><th>New Town     </th><th>New Weather  </th><th>New Town & Weather  </th></tr>
</thead>
<tbody>
<tr><td>Straight    </td><td>0.98 +/- 0.0         </td><td>0.97 +/- 0.0 </td><td>0.97 +/- 0.01</td><td>0.95 +/- 0.0        </td></tr>
<tr><td>One Turn    </td><td>0.94 +/- 0.01        </td><td>0.68 +/- 0.01</td><td>0.94 +/- 0.02</td><td>0.7 +/- 0.0         </td></tr>
<tr><td>Navigation  </td><td>0.89 +/- 0.0         </td><td>0.42 +/- 0.0 </td><td>0.83 +/- 0.01</td><td>0.46 +/- 0.02       </td></tr>
<tr><td>Nav. Dynamic</td><td>0.88 +/- 0.0         </td><td>0.42 +/- 0.02</td><td>0.8 +/- 0.03 </td><td>0.46 +/- 0.03       </td></tr>
</tbody>
</table></p><p><table>
<thead>
<tr><th>Infractions         </th><th>Training Conditions  </th><th>New Town     </th><th>New Weather  </th><th>New Town & Weather  </th></tr>
</thead>
<tbody>
<tr><td>Opposite Lane       </td><td>16.78 +/- 0.04       </td><td>2.89 +/- 0.0 </td><td>8.13 +/- 0.13</td><td>3.89 +/- 0.5        </td></tr>
<tr><td>Sidewalk            </td><td>7.46 +/- 1.32        </td><td>2.65 +/- 0.34</td><td>8.13 +/- 0.13</td><td>2.26 +/- 0.2        </td></tr>
<tr><td>Collision-static    </td><td>16.78 +/- 0.04       </td><td>8.67 +/- 0.0 </td><td>8.13 +/- 0.13</td><td>12.73 +/- 0.01      </td></tr>
<tr><td>Collision-car       </td><td>16.78 +/- 0.04       </td><td>8.67 +/- 0.0 </td><td>8.13 +/- 0.13</td><td>12.73 +/- 0.01      </td></tr>
<tr><td>Collision-pedestrian</td><td>16.78 +/- 0.04       </td><td>8.67 +/- 0.0 </td><td>8.13 +/- 0.13</td><td>12.73 +/- 0.01      </td></tr>
</tbody>
</table></p><p><table>
<thead>
<tr><th>Infractions         </th><th>Training Conditions  </th><th>New Town      </th><th>New Weather   </th><th>New Town & Weather  </th></tr>
</thead>
<tbody>
<tr><td>Opposite Lane       </td><td>13.09 +/- 2.68       </td><td>1.03 +/- 0.11 </td><td>16.86 +/- 0.09</td><td>1.35 +/- 0.1        </td></tr>
<tr><td>Sidewalk            </td><td>11.22 +/- 0.1        </td><td>0.9 +/- 0.05  </td><td>16.86 +/- 0.09</td><td>0.95 +/- 0.04       </td></tr>
<tr><td>Collision-static    </td><td>33.66 +/- 0.3        </td><td>17.68 +/- 0.26</td><td>16.86 +/- 0.09</td><td>26.88 +/- 0.7       </td></tr>
<tr><td>Collision-car       </td><td>33.66 +/- 0.3        </td><td>17.68 +/- 0.26</td><td>16.86 +/- 0.09</td><td>26.88 +/- 0.7       </td></tr>
<tr><td>Collision-pedestrian</td><td>33.66 +/- 0.3        </td><td>17.68 +/- 0.26</td><td>16.86 +/- 0.09</td><td>26.88 +/- 0.7       </td></tr>
</tbody>
</table></p><p><table>
<thead>
<tr><th>Infractions         </th><th>Training Conditions  </th><th>New Town     </th><th>New Weather   </th><th>New Town & Weather  </th></tr>
</thead>
<tbody>
<tr><td>Opposite Lane       </td><td>17.4 +/- 3.7         </td><td>2.26 +/- 0.05</td><td>23.47 +/- 8.47</td><td>3.12 +/- 0.17       </td></tr>
<tr><td>Sidewalk            </td><td>16.27 +/- 4.14       </td><td>0.62 +/- 0.03</td><td>12.71 +/- 3.75</td><td>0.68 +/- 0.03       </td></tr>
<tr><td>Collision-static    </td><td>66.63 +/- 0.2        </td><td>24.88 +/- 0.5</td><td>35.12 +/- 0.28</td><td>40.43 +/- 0.45      </td></tr>
<tr><td>Collision-car       </td><td>66.63 +/- 0.2        </td><td>24.88 +/- 0.5</td><td>35.12 +/- 0.28</td><td>40.43 +/- 0.45      </td></tr>
<tr><td>Collision-pedestrian</td><td>66.63 +/- 0.2        </td><td>24.88 +/- 0.5</td><td>35.12 +/- 0.28</td><td>40.43 +/- 0.45      </td></tr>
</tbody>
</table></p><p><table>
<thead>
<tr><th>Infractions         </th><th>Training Conditions  </th><th>New Town     </th><th>New Weather    </th><th>New Town & Weather  </th></tr>
</thead>
<tbody>
<tr><td>Opposite Lane       </td><td>20.17 +/- 2.74       </td><td>1.5 +/- 0.18 </td><td>24.95 +/- 12.95</td><td>1.66 +/- 0.04       </td></tr>
<tr><td>Sidewalk            </td><td>13.05 +/- 2.98       </td><td>0.65 +/- 0.06</td><td>4.26 +/- 0.87  </td><td>0.68 +/- 0.07       </td></tr>
<tr><td>Collision-static    </td><td>4.51 +/- 1.07        </td><td>0.38 +/- 0.05</td><td>2.15 +/- 0.55  </td><td>0.41 +/- 0.05       </td></tr>
<tr><td>Collision-car       </td><td>1.19 +/- 0.14        </td><td>0.28 +/- 0.03</td><td>1.48 +/- 0.28  </td><td>0.27 +/- 0.02       </td></tr>
<tr><td>Collision-pedestrian</td><td>14.14 +/- 5.64       </td><td>1.66 +/- 0.15</td><td>7.35 +/- 0.97  </td><td>1.83 +/- 0.14       </td></tr>
</tbody>
</table></p><p><table>
<thead>
<tr><th>Number of Infractions  </th><th>Training Conditions  </th><th>New Town     </th><th>New Weather  </th><th>New Town & Weather  </th></tr>
</thead>
<tbody>
<tr><td>Opposite Lane          </td><td>0.0 +/- 0.0          </td><td>3.0 +/- 0.0  </td><td>0.0 +/- 0.0  </td><td>3.33 +/- 0.47       </td></tr>
<tr><td>Sidewalk               </td><td>2.33 +/- 0.47        </td><td>3.33 +/- 0.47</td><td>0.0 +/- 0.0  </td><td>5.67 +/- 0.47       </td></tr>
<tr><td>Collision-static       </td><td>0.0 +/- 0.0          </td><td>0.0 +/- 0.0  </td><td>0.0 +/- 0.0  </td><td>0.0 +/- 0.0         </td></tr>
<tr><td>Collision-car          </td><td>0.0 +/- 0.0          </td><td>0.0 +/- 0.0  </td><td>0.0 +/- 0.0  </td><td>0.0 +/- 0.0         </td></tr>
<tr><td>Collision-pedestrian   </td><td>0.0 +/- 0.0          </td><td>0.0 +/- 0.0  </td><td>0.0 +/- 0.0  </td><td>0.0 +/- 0.0         </td></tr>
</tbody>
</table></p><p><table>
<thead>
<tr><th>Number of Infractions  </th><th>Training Conditions  </th><th>New Town      </th><th>New Weather  </th><th>New Town & Weather  </th></tr>
</thead>
<tbody>
<tr><td>Opposite Lane          </td><td>2.67 +/- 0.47        </td><td>17.33 +/- 1.7 </td><td>0.33 +/- 0.47</td><td>20.0 +/- 1.41       </td></tr>
<tr><td>Sidewalk               </td><td>3.0 +/- 0.0          </td><td>19.67 +/- 1.25</td><td>0.33 +/- 0.47</td><td>28.33 +/- 0.94      </td></tr>
<tr><td>Collision-static       </td><td>0.0 +/- 0.0          </td><td>0.0 +/- 0.0   </td><td>0.0 +/- 0.0  </td><td>0.0 +/- 0.0         </td></tr>
<tr><td>Collision-car          </td><td>0.0 +/- 0.0          </td><td>0.0 +/- 0.0   </td><td>0.0 +/- 0.0  </td><td>0.0 +/- 0.0         </td></tr>
<tr><td>Collision-pedestrian   </td><td>0.0 +/- 0.0          </td><td>0.0 +/- 0.0   </td><td>0.0 +/- 0.0  </td><td>0.0 +/- 0.0         </td></tr>
</tbody>
</table></p><p><table>
<thead>
<tr><th>Number of Infractions  </th><th>Training Conditions  </th><th>New Town      </th><th>New Weather  </th><th>New Town & Weather  </th></tr>
</thead>
<tbody>
<tr><td>Opposite Lane          </td><td>4.0 +/- 0.82         </td><td>11.0 +/- 0.0  </td><td>1.67 +/- 0.47</td><td>13.0 +/- 0.82       </td></tr>
<tr><td>Sidewalk               </td><td>4.33 +/- 0.94        </td><td>40.33 +/- 1.25</td><td>3.0 +/- 0.82 </td><td>59.33 +/- 1.7       </td></tr>
<tr><td>Collision-static       </td><td>0.0 +/- 0.0          </td><td>0.0 +/- 0.0   </td><td>0.0 +/- 0.0  </td><td>0.0 +/- 0.0         </td></tr>
<tr><td>Collision-car          </td><td>0.0 +/- 0.0          </td><td>0.0 +/- 0.0   </td><td>0.0 +/- 0.0  </td><td>0.0 +/- 0.0         </td></tr>
<tr><td>Collision-pedestrian   </td><td>0.0 +/- 0.0          </td><td>0.0 +/- 0.0   </td><td>0.0 +/- 0.0  </td><td>0.0 +/- 0.0         </td></tr>
</tbody>
</table></p><p><table>
<thead>
<tr><th>Number of Infractions  </th><th>Training Conditions  </th><th>New Town       </th><th>New Weather  </th><th>New Town & Weather  </th></tr>
</thead>
<tbody>
<tr><td>Opposite Lane          </td><td>3.33 +/- 0.47        </td><td>16.67 +/- 1.89 </td><td>2.33 +/- 1.89</td><td>22.67 +/- 0.47      </td></tr>
<tr><td>Sidewalk               </td><td>5.33 +/- 1.25        </td><td>38.33 +/- 3.4  </td><td>8.33 +/- 1.89</td><td>56.33 +/- 6.85      </td></tr>
<tr><td>Collision-static       </td><td>15.33 +/- 3.09       </td><td>67.33 +/- 10.4 </td><td>17.0 +/- 4.97</td><td>94.0 +/- 13.49      </td></tr>
<tr><td>Collision-car          </td><td>56.0 +/- 6.38        </td><td>90.33 +/- 11.67</td><td>24.0 +/- 5.72</td><td>139.67 +/- 8.73     </td></tr>
<tr><td>Collision-pedestrian   </td><td>5.33 +/- 1.7         </td><td>15.0 +/- 1.41  </td><td>4.67 +/- 0.47</td><td>20.67 +/- 1.7       </td></tr>
</tbody>
</table></p></body>
</html>
