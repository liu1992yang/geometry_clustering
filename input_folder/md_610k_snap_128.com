%mem=64gb
%nproc=28       
%Chk=snap_128.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_128 

2     1 
  O    1.524825941  -4.287725078  -5.774827726
  C    2.034031976  -3.071792323  -6.342050874
  H    3.054575842  -3.216385666  -6.744505746
  H    2.060172899  -2.354747817  -5.493990119
  C    1.062645970  -2.589783129  -7.423958727
  H    1.188245237  -1.508867318  -7.659854338
  O    1.384649674  -3.251492530  -8.693892389
  C    0.312303251  -4.095083368  -9.098129738
  H    0.323750560  -4.111305048 -10.214476364
  C   -0.406774095  -2.977164801  -7.107520549
  H   -0.454035193  -3.716846663  -6.261041793
  C   -0.946332113  -3.566917376  -8.412919955
  H   -1.435992729  -2.779311666  -9.029728203
  H   -1.768258800  -4.306599165  -8.257428384
  O   -1.155477662  -1.765290842  -6.816546013
  N    0.717992783  -5.478975425  -8.609626499
  C   -0.085940766  -6.506246903  -8.087414351
  C    0.791974183  -7.546536451  -7.660433521
  N    2.110255333  -7.120019448  -7.879234575
  C    2.061836652  -5.880916046  -8.428004574
  N   -1.444791618  -6.534541974  -7.973094908
  C   -1.970891254  -7.687765998  -7.441248876
  N   -1.172616266  -8.787112993  -7.048225117
  C    0.283348694  -8.788779147  -7.128021867
  N   -3.323537793  -7.701004746  -7.228871430
  H   -3.860926664  -6.857691489  -7.487199233
  H   -3.845463290  -8.547902544  -7.051004839
  O    0.866377946  -9.781046143  -6.764206785
  H    2.915620398  -5.252196962  -8.707957207
  H   -1.607094189  -9.645754537  -6.676614020
  P   -1.366058069  -1.542497921  -5.204810203
  O   -0.830496976   0.033644564  -5.101707545
  C   -0.218814833   0.412705437  -3.843716606
  H    0.669325633   1.004801035  -4.174182952
  H    0.128207392  -0.477974327  -3.269616649
  C   -1.128098706   1.295128893  -2.989001457
  H   -0.739636239   1.306313307  -1.930625810
  O   -0.924468968   2.660578489  -3.470681486
  C   -2.135987346   3.421571519  -3.401182278
  H   -1.938373920   4.259716327  -2.683936629
  C   -2.661032632   1.062307773  -2.978092173
  H   -3.046326957   0.424367277  -3.807794599
  C   -3.272249134   2.471797980  -3.009107658
  H   -3.673767003   2.714723097  -1.991918407
  H   -4.157523366   2.533667948  -3.669983334
  O   -3.048895806   0.536209806  -1.693266809
  O   -0.661300478  -2.485017926  -4.308000932
  O   -2.923899235  -1.308199321  -5.015458738
  N   -2.332684555   4.020011084  -4.755039781
  C   -2.310665442   3.401093101  -6.001023839
  C   -2.584168014   4.419039684  -6.973729568
  N   -2.757013510   5.652481574  -6.327780859
  C   -2.602450523   5.418033039  -5.027109037
  N   -2.104132678   2.082139885  -6.387584464
  C   -2.167492910   1.815117350  -7.762784015
  N   -2.422650956   2.720011046  -8.698885322
  C   -2.636663489   4.072695895  -8.360810943
  H   -2.003332950   0.757876726  -8.072517251
  N   -2.878786491   4.949078898  -9.350601063
  H   -2.910843524   4.656123207 -10.325629693
  H   -3.036932445   5.937205953  -9.149719985
  H   -2.664654730   6.152099448  -4.226422939
  P   -3.007630812  -1.106682264  -1.532282727
  O   -1.377687271  -1.180947582  -1.478068700
  C   -0.703016916  -2.453864714  -1.350030141
  H   -0.575470963  -2.866827825  -2.384128476
  H    0.291939430  -2.168274683  -0.941392589
  C   -1.460430022  -3.400913831  -0.431510282
  H   -1.474304140  -3.088082191   0.641429979
  O   -2.861949009  -3.311462023  -0.868399369
  C   -3.327374301  -4.596899358  -1.401777829
  H   -4.371714967  -4.651951009  -1.002054148
  C   -1.043859447  -4.882068390  -0.581863669
  H   -0.286845114  -5.042935539  -1.400593357
  C   -2.366799477  -5.635280910  -0.833427323
  H   -2.215515794  -6.516928443  -1.491350308
  H   -2.762039960  -6.053458053   0.122448531
  O   -0.486127282  -5.214813543   0.694955300
  O   -3.738163501  -1.791294796  -2.661901883
  O   -3.656724454  -1.208921249  -0.060978370
  N   -3.390069322  -4.540898060  -2.889020119
  C   -2.318457437  -4.981385809  -3.731806654
  N   -2.583753434  -4.962659116  -5.124251599
  C   -3.801613094  -4.504131638  -5.740275316
  C   -4.757973640  -3.912738020  -4.804817434
  C   -4.546202202  -3.948538659  -3.464256924
  O   -1.259535229  -5.411264268  -3.317457110
  H   -1.857780212  -5.365365425  -5.733730233
  O   -3.892645724  -4.674651910  -6.940852835
  C   -5.966421984  -3.268289442  -5.391380732
  H   -6.896948934  -3.752262346  -5.058267068
  H   -6.028787697  -2.202929309  -5.119592407
  H   -5.962762661  -3.313792666  -6.494108965
  H   -5.269150791  -3.511561975  -2.755816024
  P    0.374925234  -6.625324057   0.796838556
  O    0.846242036  -6.836747793  -0.767074495
  C    2.245644906  -6.935057079  -1.083927991
  H    2.839556065  -7.403893595  -0.272323083
  H    2.239997324  -7.613623865  -1.962125448
  C    2.800576847  -5.544332756  -1.422382910
  H    2.757985662  -4.838130054  -0.561377663
  O    4.233053202  -5.785685607  -1.650304670
  C    4.611966557  -5.374854133  -2.974182192
  H    5.666432309  -5.028478078  -2.890582085
  C    2.274641955  -4.902070944  -2.731309384
  H    1.733938219  -5.624490648  -3.381057041
  C    3.562871390  -4.352381812  -3.382438495
  H    3.477365391  -4.190242246  -4.471788928
  H    3.801273307  -3.352581931  -2.962490343
  O    1.386282040  -3.832631419  -2.413992383
  O   -0.335024391  -7.732544256   1.421530897
  O    1.751848099  -6.078642298   1.494991663
  N    4.567564522  -6.638463631  -3.819945308
  C    4.016620943  -6.695979515  -5.159052278
  N    3.680861451  -7.926772400  -5.696123742
  C    3.990785009  -9.087961232  -5.030573400
  C    4.718872977  -9.043865765  -3.786195142
  C    4.957333448  -7.831408925  -3.201194979
  O    3.823838099  -5.667102787  -5.806609530
  N    3.571979416 -10.255755005  -5.594930890
  H    3.736253018 -11.148276597  -5.150659726
  H    2.909898319 -10.245676892  -6.369948746
  H    2.128522997  -5.049923179  -5.990646950
  H    5.442064027  -7.739117827  -2.212700458
  H    5.068067740  -9.965563938  -3.319201042
  H    0.831445312  -3.650491888  -3.237041825
  H   -3.378911957  -1.749572194  -4.075111763
  H   -3.958311010  -2.128764483   0.250623392
  H    1.859868910  -6.191090352   2.479744026
  H    2.940669303  -7.604339387  -7.445052812
  H   -1.798071360   1.355964364  -5.696575303
