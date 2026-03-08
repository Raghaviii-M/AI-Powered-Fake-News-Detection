let chart

function checkNews(){

let news=document.getElementById("newsText").value

fetch("http://127.0.0.1:5000/predict",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({
news:news
})

})

.then(res=>res.json())

.then(data=>{

document.getElementById("prediction").innerHTML =
"Prediction: "+data.prediction

// probability meter
document.getElementById("fakeMeter").style.width =
data.fake_probability+"%"

document.getElementById("fakePercent").innerHTML =
"Fake Probability: "+data.fake_probability+"%"

// explanation
document.getElementById("explanation").innerHTML =
data.explanation

createChart(data.real_probability,data.fake_probability)

})

}

function createChart(real,fake){

let ctx=document.getElementById("resultChart").getContext("2d")

if(chart){
chart.destroy()
}

chart=new Chart(ctx,{

type:"pie",

data:{
labels:["Real News","Fake News"],
datasets:[{
data:[real,fake],
backgroundColor:["#4CAF50","#ff5252"]
}]
}

})

}