Option Explicit

Public Const OVERALL_MEAN As Double = 4525.5798898072
Public Const OVERALL_STD As Double = 411.2093264513

Private gStatsInitialized As Boolean

Public dowMean(0 To 6) As Double
Public dowStd(0 To 6) As Double
Public monthMean(0 To 11) As Double
Public monthStd(0 To 11) As Double
Public holidayMean(0 To 9) As Double
Public holidayStd(0 To 9) As Double
Public cnyOffsetMean(-25 To 15) As Double

Private Sub Init_dowMean()
    dowMean(0) = 4535.3142857143
    dowMean(1) = 4414.6952380952
    dowMean(2) = 4406.5145631068
    dowMean(3) = 4394.1067961165
    dowMean(4) = 4664.9514563107
    dowMean(5) = 4611.2718446602
    dowMean(6) = 4652.9326923077
End Sub

Private Sub Init_dowStd()
    dowStd(0) = 398.4926481873
    dowStd(1) = 440.1042517188
    dowStd(2) = 448.9279138622
    dowStd(3) = 475.6838815321
    dowStd(4) = 359.9063434448
    dowStd(5) = 297.0685599277
    dowStd(6) = 328.2469844558
End Sub

Private Sub Init_monthMean()
    monthMean(0) = 4732.8225806452
    monthMean(1) = 4774.2982456140
    monthMean(2) = 4346.9032258065
    monthMean(3) = 4489.9000000000
    monthMean(4) = 4467.9677419355
    monthMean(5) = 4474.9333333333
    monthMean(6) = 4526.8225806452
    monthMean(7) = 4677.1403508772
    monthMean(8) = 4511.1500000000
    monthMean(9) = 4524.6935483871
    monthMean(10) = 4412.9000000000
    monthMean(11) = 4392.8225806452
End Sub

Private Sub Init_monthStd()
    monthStd(0) = 568.2186004017
    monthStd(1) = 657.3723929650
    monthStd(2) = 223.1521358337
    monthStd(3) = 391.5435642580
    monthStd(4) = 424.4209157151
    monthStd(5) = 349.9327715902
    monthStd(6) = 125.7204189623
    monthStd(7) = 130.9966518924
    monthStd(8) = 485.7039212859
    monthStd(9) = 438.3750766485
    monthStd(10) = 241.7097783482
    monthStd(11) = 303.2044445905
End Sub

Private Sub Init_holidayMean()
    holidayMean(0) = 4406.1721991701
    holidayMean(1) = 4592.8265895954
    holidayMean(2) = 4586.2000000000
    holidayMean(3) = 5545.4666666667
    holidayMean(4) = 4985.0000000000
    holidayMean(5) = 5403.4000000000
    holidayMean(6) = 5263.5000000000
    holidayMean(7) = 5775.5000000000
    holidayMean(8) = 5233.2142857143
    holidayMean(9) = 4648.6000000000
End Sub

Private Sub Init_holidayStd()
    holidayStd(0) = 317.5286806090
    holidayStd(1) = 206.7052276778
    holidayStd(2) = 502.8366533975
    holidayStd(3) = 751.4373718478
    holidayStd(4) = 472.3677945556
    holidayStd(5) = 451.0169250335
    holidayStd(6) = 573.3396317483
    holidayStd(7) = 418.6012422342
    holidayStd(8) = 391.0083962450
    holidayStd(9) = 556.8006568141
End Sub

Private Sub Init_cnyOffsetMean()
    cnyOffsetMean(-25) = 4261.0000000000
    cnyOffsetMean(-24) = 4145.0000000000
    cnyOffsetMean(-23) = 4298.0000000000
    cnyOffsetMean(-22) = 4462.0000000000
    cnyOffsetMean(-21) = 4218.5000000000
    cnyOffsetMean(-20) = 4258.5000000000
    cnyOffsetMean(-19) = 4270.0000000000
    cnyOffsetMean(-18) = 4245.0000000000
    cnyOffsetMean(-17) = 4323.5000000000
    cnyOffsetMean(-16) = 4478.5000000000
    cnyOffsetMean(-15) = 4747.5000000000
    cnyOffsetMean(-14) = 4732.0000000000
    cnyOffsetMean(-13) = 4820.5000000000
    cnyOffsetMean(-12) = 4832.0000000000
    cnyOffsetMean(-11) = 5024.0000000000
    cnyOffsetMean(-10) = 4991.0000000000
    cnyOffsetMean(-9) = 4992.5000000000
    cnyOffsetMean(-8) = 4945.5000000000
    cnyOffsetMean(-7) = 4918.5000000000
    cnyOffsetMean(-6) = 4987.5000000000
    cnyOffsetMean(-5) = 5394.0000000000
    cnyOffsetMean(-4) = 5576.0000000000
    cnyOffsetMean(-3) = 5854.5000000000
    cnyOffsetMean(-2) = 5806.0000000000
    cnyOffsetMean(-1) = 4464.5000000000
    cnyOffsetMean(0) = 4266.0000000000
    cnyOffsetMean(1) = 5515.0000000000
    cnyOffsetMean(2) = 5698.0000000000
    cnyOffsetMean(3) = 5851.5000000000
    cnyOffsetMean(4) = 5939.0000000000
    cnyOffsetMean(5) = 6311.0000000000
    cnyOffsetMean(6) = 5813.5000000000
    cnyOffsetMean(7) = 5815.5000000000
    cnyOffsetMean(8) = 5402.5000000000
    cnyOffsetMean(9) = 5081.0000000000
    cnyOffsetMean(10) = 4606.5000000000
    cnyOffsetMean(11) = 4340.0000000000
    cnyOffsetMean(12) = 4264.0000000000
    cnyOffsetMean(13) = 4482.0000000000
    cnyOffsetMean(14) = 4776.5000000000
    cnyOffsetMean(15) = 5283.5000000000
End Sub

Public Sub EnsureStatsInitialized()
    If gStatsInitialized Then Exit Sub
    Init_dowMean
    Init_dowStd
    Init_monthMean
    Init_monthStd
    Init_holidayMean
    Init_holidayStd
    Init_cnyOffsetMean
    gStatsInitialized = True
End Sub
