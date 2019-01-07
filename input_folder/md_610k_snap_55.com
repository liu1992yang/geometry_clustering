%mem=64gb
%nproc=28       
%Chk=snap_55.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_55 

2     1 
  O    1.827419930  -6.340992673  -6.654372575
  C    2.699105533  -5.263245488  -7.026942346
  H    3.256018320  -5.671082409  -7.899634703
  H    3.401216516  -5.065571608  -6.193435694
  C    1.948489282  -3.990542751  -7.435088069
  H    2.636077304  -3.105812610  -7.439255735
  O    1.563902204  -4.112978224  -8.844122011
  C    0.156402958  -4.101513627  -9.004887283
  H   -0.038732400  -3.509549352  -9.933556476
  C    0.655826090  -3.677583470  -6.647455565
  H    0.457383180  -4.438651377  -5.855885661
  C   -0.466585562  -3.596369180  -7.701702471
  H   -0.823514623  -2.551515944  -7.809832143
  H   -1.358334380  -4.180301175  -7.389042931
  O    0.888842730  -2.378546963  -6.073081184
  N   -0.271860358  -5.523300435  -9.284065349
  C   -0.863970944  -6.017842651 -10.466433675
  C   -1.019711731  -7.421073149 -10.288328354
  N   -0.512442548  -7.752789839  -9.013397773
  C   -0.065534171  -6.615037214  -8.427682219
  N   -1.232269574  -5.311720105 -11.579049799
  C   -1.768660821  -6.076537379 -12.588144784
  N   -1.945733926  -7.474171933 -12.497472669
  C   -1.583468286  -8.258933549 -11.317351252
  N   -2.174724162  -5.424223859 -13.726305983
  H   -2.001707720  -4.426094217 -13.816983163
  H   -2.480896855  -5.912553082 -14.555352524
  O   -1.787510790  -9.444199261 -11.346121883
  H    0.464995010  -6.502551177  -7.402228846
  H   -2.352816872  -8.006820692 -13.280856567
  P    0.052912184  -1.961362043  -4.699445481
  O   -1.353859481  -1.415537574  -5.441260189
  C   -1.323247949  -0.006500559  -5.729124648
  H   -2.184040472   0.124078794  -6.416163967
  H   -0.395819247   0.278373983  -6.265727515
  C   -1.491092590   0.796287246  -4.427363125
  H   -0.597533978   0.706991455  -3.738906362
  O   -1.483294812   2.219086419  -4.768917050
  C   -2.699673517   2.861648654  -4.382005276
  H   -2.391537510   3.738190558  -3.752817463
  C   -2.832429664   0.503098872  -3.716645708
  H   -3.409016132  -0.321580767  -4.197051958
  C   -3.600627864   1.838618883  -3.696303448
  H   -3.788949123   2.121966074  -2.628529862
  H   -4.603281674   1.734024460  -4.146864790
  O   -2.583537753   0.218995832  -2.329101018
  O    0.751559616  -0.928177885  -3.923106041
  O   -0.176939816  -3.478692892  -4.162720005
  N   -3.275825622   3.399301871  -5.665458121
  C   -2.547878023   3.687866741  -6.814027271
  C   -3.455120993   4.297126218  -7.741165154
  N   -4.726505364   4.386032451  -7.157625391
  C   -4.616035882   3.864868634  -5.934661256
  N   -1.204653396   3.488584285  -7.131573389
  C   -0.796892878   3.937633878  -8.398390039
  N   -1.598405454   4.502110449  -9.289475474
  C   -2.964565823   4.728453627  -9.015101911
  H    0.275637876   3.800011118  -8.663913570
  N   -3.711241610   5.330193200  -9.954603825
  H   -3.310633714   5.632441196 -10.843090050
  H   -4.699252501   5.529025388  -9.792136431
  H   -5.408247093   3.794906958  -5.193295473
  P   -2.297237353  -1.363021474  -1.941477391
  O   -1.629987642  -1.047440137  -0.500413734
  C   -0.728308365  -2.068221358   0.019851553
  H    0.062627315  -2.317295905  -0.711779698
  H   -0.261080847  -1.540842517   0.888408247
  C   -1.624078464  -3.225230278   0.447401573
  H   -2.259835478  -2.954507571   1.325104725
  O   -2.555047732  -3.362428015  -0.686807955
  C   -2.850876730  -4.775457661  -0.942341841
  H   -3.967400974  -4.822203566  -0.891255701
  C   -0.997616998  -4.629358931   0.617975291
  H   -0.057517119  -4.751147615   0.017921560
  C   -2.127332373  -5.565584256   0.141491070
  H   -1.740329814  -6.554398483  -0.179114999
  H   -2.799454035  -5.800851815   1.000032608
  O   -0.807447552  -4.966089565   1.986971076
  O   -1.531989567  -2.086703701  -3.002174491
  O   -3.854164678  -1.860050752  -1.826095912
  N   -2.426153929  -5.081933244  -2.335901131
  C   -1.211819608  -5.783740885  -2.641015405
  N   -1.049323773  -6.180549178  -3.983699145
  C   -1.873087167  -5.751107921  -5.071893249
  C   -3.014583461  -4.909414077  -4.687702139
  C   -3.253659454  -4.607942813  -3.387711407
  O   -0.365311432  -6.025367352  -1.795761132
  H   -0.179211380  -6.710732811  -4.201579712
  O   -1.541661268  -6.101665097  -6.192783098
  C   -3.857972590  -4.367048899  -5.788656536
  H   -3.568999029  -3.329340116  -6.024819704
  H   -3.758377096  -4.961400051  -6.711821337
  H   -4.928985929  -4.367765325  -5.537063913
  H   -4.092257838  -3.960639249  -3.082997995
  P    0.539649145  -4.431592090   2.796077302
  O    1.737133816  -5.524567172   2.541332962
  C    1.989900231  -6.206798053   1.308273505
  H    2.436089829  -7.167624181   1.663981294
  H    1.075966291  -6.442040913   0.737975287
  C    3.040839744  -5.397301352   0.534316345
  H    3.743162193  -4.905019988   1.252308080
  O    3.889066132  -6.330991991  -0.188867069
  C    3.794131609  -6.134978077  -1.609167072
  H    4.833046579  -6.288236252  -1.989995591
  C    2.473836543  -4.413292960  -0.520480495
  H    1.356930141  -4.478444186  -0.628853981
  C    3.202879970  -4.741740569  -1.824375457
  H    2.525088573  -4.668554643  -2.710125863
  H    3.998630699  -4.005975346  -2.056463347
  O    2.662699175  -3.066520442  -0.088228544
  O    0.322490556  -4.157581119   4.200806788
  O    0.888309290  -3.194681102   1.770688468
  N    2.930558983  -7.266951233  -2.117503064
  C    2.160463719  -7.168007289  -3.340922942
  N    1.341823950  -8.216728233  -3.716027464
  C    1.297828423  -9.356977332  -2.958835803
  C    2.106893309  -9.500284838  -1.774100769
  C    2.892869526  -8.452201702  -1.381191672
  O    2.246001916  -6.142520850  -4.020738512
  N    0.437207269 -10.342068375  -3.361986755
  H    0.412853601 -11.240679647  -2.903349107
  H   -0.087775851 -10.250058306  -4.221204487
  H    1.699705331  -6.395903120  -5.651030667
  H    3.526693468  -8.498051171  -0.476860880
  H    2.098904811 -10.430501946  -1.206566169
  H    3.609349264  -2.788086806  -0.115603873
  H   -0.737601481  -3.565436263  -3.306506606
  H   -4.222581345  -2.018645484  -0.907643373
  H    1.828762356  -2.996321251   1.468488500
  H   -0.500392101  -8.694097303  -8.636830307
  H   -0.575092383   3.068193070  -6.425196795

