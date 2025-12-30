Option Explicit

Private gCalendar As Object
Private gCalendarCols As Object
Private gFeatureIdx As Object
Private gInitialized As Boolean

Private Const SHEET_DATA As String = "Data"
Private Const SHEET_CALENDAR As String = "Calendar"
Private Const SHEET_FEATURES As String = "FeatureMap"
Private Const COL_DATE As Long = 1
Private Const COL_ACTUAL As Long = 2
Private Const COL_PRED As Long = 3
Private Const FIRST_DATA_ROW As Long = 2
Private Const FORCE_ALL As Boolean = False

Public Sub PredictAll()
    EnsureInitialized
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets(SHEET_DATA)

    Dim lastRow As Long
    lastRow = ws.Cells(ws.Rows.Count, COL_DATE).End(xlUp).Row

    Dim r As Long
    For r = FIRST_DATA_ROW To lastRow
        If Not HasHistory(r, ws) Then GoTo NextRow
        If (Not FORCE_ALL) Then
            If Len(ws.Cells(r, COL_ACTUAL).Value) > 0 Then GoTo NextRow
        End If
        Dim feats() As Double
        feats = BuildFeatureVector(r, ws)
        ws.Cells(r, COL_PRED).Value = XGBPredict(feats)
NextRow:
    Next r
End Sub

Public Function PredictAtRow(ByVal rowNum As Long) As Double
    EnsureInitialized
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets(SHEET_DATA)
    If Not HasHistory(rowNum, ws) Then
        PredictAtRow = CVErr(xlErrNA)
        Exit Function
    End If
    Dim feats() As Double
    feats = BuildFeatureVector(rowNum, ws)
    PredictAtRow = XGBPredict(feats)
End Function

Public Function PredictByDate(ByVal targetDate As Date) As Double
    EnsureInitialized
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets(SHEET_DATA)

    Dim lastRow As Long
    lastRow = ws.Cells(ws.Rows.Count, COL_DATE).End(xlUp).Row

    Dim r As Long
    For r = FIRST_DATA_ROW To lastRow
        If CLng(CDate(ws.Cells(r, COL_DATE).Value)) = CLng(targetDate) Then
            PredictByDate = PredictAtRow(r)
            Exit Function
        End If
    Next r

    PredictByDate = CVErr(xlErrNA)
End Function

Private Sub EnsureInitialized()
    If gInitialized Then Exit Sub
    EnsureStatsInitialized
    LoadCalendar
    LoadFeatureMap
    gInitialized = True
End Sub

Private Sub LoadFeatureMap()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets(SHEET_FEATURES)
    Dim data As Variant
    data = ws.Range("A1").CurrentRegion.Value

    Dim dict As Object
    Set dict = CreateObject("Scripting.Dictionary")

    Dim i As Long
    For i = 2 To UBound(data, 1)
        dict(CStr(data(i, 2))) = CLng(data(i, 1))
    Next i

    Set gFeatureIdx = dict
End Sub

Private Sub LoadCalendar()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets(SHEET_CALENDAR)
    Dim data As Variant
    data = ws.Range("A1").CurrentRegion.Value

    Dim cols As Object
    Set cols = CreateObject("Scripting.Dictionary")

    Dim c As Long
    For c = 1 To UBound(data, 2)
        cols(CStr(data(1, c))) = c
    Next c

    Dim dict As Object
    Set dict = CreateObject("Scripting.Dictionary")

    Dim i As Long
    For i = 2 To UBound(data, 1)
        Dim d As Double
        d = CLng(CDate(data(i, cols("date"))))
        dict(d) = Application.Index(data, i, 0)
    Next i

    Set gCalendar = dict
    Set gCalendarCols = cols
End Sub

Private Function GetCalendarValue(ByVal dateKey As Double, ByVal colName As String) As Double
    Dim rowData As Variant
    rowData = gCalendar(dateKey)
    GetCalendarValue = CDbl(rowData(gCalendarCols(colName)))
End Function

Private Function BuildFeatureVector(ByVal r As Long, ByVal ws As Worksheet) As Double()
    Dim feats() As Double
    ReDim feats(0 To gFeatureIdx.Count - 1)

    Dim dateKey As Double
    dateKey = CLng(CDate(ws.Cells(r, COL_DATE).Value))
    If Not gCalendar.Exists(dateKey) Then
        Err.Raise vbObjectError + 1, , "Date not found in Calendar sheet."
    End If

    Dim y1 As Double, y7 As Double, y14 As Double, y21 As Double, y28 As Double
    y1 = GetValue(r - 1, ws)
    y7 = GetValue(r - 7, ws)
    y14 = GetValue(r - 14, ws)
    y21 = GetValue(r - 21, ws)
    y28 = GetValue(r - 28, ws)

    Dim roll7 As Double, roll14 As Double, roll30 As Double
    Dim std7 As Double, std14 As Double
    roll7 = MeanWindow(r, 7, ws)
    roll14 = MeanWindow(r, 14, ws)
    roll30 = MeanWindow(r, 30, ws)
    std7 = StdWindow(r, 7, ws)
    std14 = StdWindow(r, 14, ws)

    Dim trend7 As Double
    trend7 = roll7 - MeanWindow(r - 7, 7, ws)
    Dim recentChange As Double
    recentChange = (y1 - GetValue(r - 4, ws)) / 3#

    Dim deltaRoll7 As Double, deltaLag7 As Double
    deltaRoll7 = y1 - roll7
    deltaLag7 = y1 - y7

    Dim dow As Long, month As Long, htype As Long, daysToCny As Long
    dow = CLng(GetCalendarValue(dateKey, "day_of_week"))
    month = CLng(GetCalendarValue(dateKey, "month"))
    htype = CLng(GetCalendarValue(dateKey, "holiday_type"))
    daysToCny = CLng(GetCalendarValue(dateKey, "days_to_cny"))

    ' Base calendar features
    SetFeature feats, "year", GetCalendarValue(dateKey, "year")
    SetFeature feats, "month", GetCalendarValue(dateKey, "month")
    SetFeature feats, "day", GetCalendarValue(dateKey, "day")
    SetFeature feats, "day_of_week", GetCalendarValue(dateKey, "day_of_week")
    SetFeature feats, "day_of_year", GetCalendarValue(dateKey, "day_of_year")
    SetFeature feats, "is_weekend", GetCalendarValue(dateKey, "is_weekend")
    SetFeature feats, "dow_sin", GetCalendarValue(dateKey, "dow_sin")
    SetFeature feats, "dow_cos", GetCalendarValue(dateKey, "dow_cos")
    SetFeature feats, "month_sin", GetCalendarValue(dateKey, "month_sin")
    SetFeature feats, "month_cos", GetCalendarValue(dateKey, "month_cos")
    SetFeature feats, "doy_sin", GetCalendarValue(dateKey, "doy_sin")
    SetFeature feats, "doy_cos", GetCalendarValue(dateKey, "doy_cos")
    SetFeature feats, "is_holiday", GetCalendarValue(dateKey, "is_holiday")
    SetFeature feats, "is_statutory_holiday", GetCalendarValue(dateKey, "is_statutory_holiday")
    SetFeature feats, "holiday_type", GetCalendarValue(dateKey, "holiday_type")
    SetFeature feats, "is_adjusted_workday", GetCalendarValue(dateKey, "is_adjusted_workday")
    SetFeature feats, "days_to_next_holiday", GetCalendarValue(dateKey, "days_to_next_holiday")
    SetFeature feats, "next_holiday_type", GetCalendarValue(dateKey, "next_holiday_type")
    SetFeature feats, "days_from_prev_holiday", GetCalendarValue(dateKey, "days_from_prev_holiday")
    SetFeature feats, "prev_holiday_type", GetCalendarValue(dateKey, "prev_holiday_type")
    SetFeature feats, "days_to_nearest_holiday", GetCalendarValue(dateKey, "days_to_nearest_holiday")
    SetFeature feats, "holiday_proximity", GetCalendarValue(dateKey, "holiday_proximity")
    SetFeature feats, "holiday_phase", GetCalendarValue(dateKey, "holiday_phase")
    SetFeature feats, "holiday_day_num", GetCalendarValue(dateKey, "holiday_day_num")
    SetFeature feats, "total_holiday_length", GetCalendarValue(dateKey, "total_holiday_length")
    SetFeature feats, "holiday_progress", GetCalendarValue(dateKey, "holiday_progress")
    SetFeature feats, "days_to_cny", GetCalendarValue(dateKey, "days_to_cny")
    SetFeature feats, "cny_window", GetCalendarValue(dateKey, "cny_window")
    SetFeature feats, "cny_pre", GetCalendarValue(dateKey, "cny_pre")
    SetFeature feats, "cny_post", GetCalendarValue(dateKey, "cny_post")
    SetFeature feats, "cny_day", GetCalendarValue(dateKey, "cny_day")
    SetFeature feats, "year_normalized", GetCalendarValue(dateKey, "year_normalized")

    ' Group stats from training data
    SetFeature feats, "dow_mean", dowMean(dow)
    SetFeature feats, "dow_std", dowStd(dow)
    SetFeature feats, "month_mean", monthMean(month)
    SetFeature feats, "month_std", monthStd(month)
    SetFeature feats, "holiday_type_mean", holidayMean(htype)
    SetFeature feats, "holiday_type_std", holidayStd(htype)
    If daysToCny >= -25 And daysToCny <= 15 Then
        SetFeature feats, "cny_offset_mean", cnyOffsetMean(daysToCny)
    Else
        SetFeature feats, "cny_offset_mean", OVERALL_MEAN
    End If

    ' Dynamic features
    SetFeature feats, "lag_1", y1
    SetFeature feats, "lag_2", GetValue(r - 2, ws)
    SetFeature feats, "lag_3", GetValue(r - 3, ws)
    SetFeature feats, "lag_7", y7
    SetFeature feats, "lag_14", y14
    SetFeature feats, "lag_21", y21
    SetFeature feats, "lag_28", y28
    SetFeature feats, "roll_7", roll7
    SetFeature feats, "roll_14", roll14
    SetFeature feats, "roll_30", roll30
    SetFeature feats, "std_7", std7
    SetFeature feats, "std_14", std14
    SetFeature feats, "trend_7", trend7
    SetFeature feats, "recent_change_3d", recentChange
    SetFeature feats, "delta_vs_roll7", deltaRoll7
    SetFeature feats, "delta_vs_lag7", deltaLag7

    BuildFeatureVector = feats
End Function

Private Sub SetFeature(ByRef feats() As Double, ByVal name As String, ByVal value As Double)
    If gFeatureIdx.Exists(name) Then
        feats(gFeatureIdx(name)) = value
    End If
End Sub

Private Function GetValue(ByVal r As Long, ByVal ws As Worksheet) As Double
    Dim v As Variant
    v = ws.Cells(r, COL_ACTUAL).Value
    If IsNumeric(v) And Len(v) > 0 Then
        GetValue = CDbl(v)
        Exit Function
    End If
    v = ws.Cells(r, COL_PRED).Value
    If IsNumeric(v) And Len(v) > 0 Then
        GetValue = CDbl(v)
        Exit Function
    End If
    Err.Raise vbObjectError + 2, , "Missing history value."
End Function

Private Function MeanWindow(ByVal endRow As Long, ByVal window As Long, ByVal ws As Worksheet) As Double
    Dim i As Long
    Dim s As Double
    s = 0#
    For i = endRow - window To endRow - 1
        s = s + GetValue(i, ws)
    Next i
    MeanWindow = s / window
End Function

Private Function StdWindow(ByVal endRow As Long, ByVal window As Long, ByVal ws As Worksheet) As Double
    Dim i As Long
    Dim m As Double, s As Double
    m = MeanWindow(endRow, window, ws)
    s = 0#
    For i = endRow - window To endRow - 1
        s = s + (GetValue(i, ws) - m) ^ 2
    Next i
    StdWindow = Sqr(s / (window - 1))
End Function

Private Function HasHistory(ByVal r As Long, ByVal ws As Worksheet) As Boolean
    Dim i As Long
    If (r - 30) < FIRST_DATA_ROW Then
        HasHistory = False
        Exit Function
    End If
    For i = r - 30 To r - 1
        If Not HasNumericValue(i, ws) Then
            HasHistory = False
            Exit Function
        End If
    Next i
    HasHistory = True
End Function

Private Function HasNumericValue(ByVal r As Long, ByVal ws As Worksheet) As Boolean
    Dim v As Variant
    v = ws.Cells(r, COL_ACTUAL).Value
    If IsNumeric(v) And Len(v) > 0 Then
        HasNumericValue = True
        Exit Function
    End If
    v = ws.Cells(r, COL_PRED).Value
    If IsNumeric(v) And Len(v) > 0 Then
        HasNumericValue = True
        Exit Function
    End If
    HasNumericValue = False
End Function
