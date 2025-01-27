using LottoML_net6;

//Load sample data
var sampleData = new _655_slot1.ModelInput()
{
    Date = @"8/31/2022",
    Results = @"15-12-16-17-03-09",
    Slot_1 = 3F,
    Slot_2 = 9F,
    Slot_3 = 12F,
    Slot_4 = 15F,
    Slot_5 = 16F,
    Slot_6 = 17F,
    Sorted2 = 9F,
    Sorted3 = 12F,
    Sorted4 = 15F,
    Sorted5 = 16F,
    Sorted6 = 17F,
};

//Load model and predict output
var result = _655_slot1.Predict(sampleData);
Console.WriteLine(result.Score);
