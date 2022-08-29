using LottoML_net6;

var slot1 = _655_slot1.Predict();
var slot2 = _655_slot2.Predict();
var slot3 = _655_slot3.Predict();
var slot4 = _655_slot4.Predict();
var slot5 = _655_slot5.Predict();
var slot6 = _655_slot6.Predict();

for(var i=0; i<5; i++)
{
    var combination = $"{Math.Floor(slot1.Sorted1[i])}-{Math.Floor(slot2.Sorted2[i])}-{Math.Floor(slot3.Sorted3[i])}-{Math.Floor(slot4.Sorted4[i])}-{Math.Floor(slot5.Sorted5[i])}-{Math.Floor(slot6.Sorted6[i])}";
    Console.WriteLine(combination);
}