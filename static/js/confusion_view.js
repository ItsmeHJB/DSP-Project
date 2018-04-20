d3.selection.prototype.moveToFront = function() {  
      return this.each(function(){
        this.parentNode.appendChild(this);
      });
    };
d3.selection.prototype.moveToBack = function() {  
  return this.each(function() { 
      var firstChild = this.parentNode.firstChild; 
      if (firstChild) { 
          this.parentNode.insertBefore(this, firstChild); 
      } 
  });
};



var marginScatter = { top: 25, right: 50, bottom: 50, left: 30 },
    fullWidthScatter = 425,
    fullHeightScatter = 380,
    widthScatter = fullWidthScatter - marginScatter.left - marginScatter.right,
    heightScatter = fullHeightScatter - marginScatter.top - marginScatter.bottom;

var confusion_colormap = ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d']
//var confusion_colormap = ['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#08519c','#08306b']
//var confusion_colormap = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']

//ar colorScale = d3.scaleLinear()
    //.range(["#2c7bb6", "#00a6ca","#00ccbc","#90eb9d","#ffff8c","#f9d057","#f29e2e","#e76818","#d7191c"]);


var xScaleScatter = d3.scaleLinear()
  .domain([-1 , 10])
  .range([0 , widthScatter]);

var yScaleScatter = d3.scaleLinear()
  .domain([-1 , 10])
  .range([heightScatter, 0]);

var scatter_data;

function create_confusion_plot(placement, data) {

  var numrows = 10;
  var numcols = 10;

  var max = 0

  for (i in data) {
    if (max < data[i]['value']) {
      max = data[i]['value'];
    }
  }

  scatter_data = data


  console.log(scatter_data);


  var canvas = d3.select("#scatter_plot_canvas")
    .attr('width', widthScatter - 1)
    .attr('height', heightScatter - 1)
    .style("transform", "translate(" + (marginScatter.left - fullWidthScatter ) +
            "px" + "," + (marginScatter.top + 1) + "px" + ")");

  var context = canvas.node().getContext('2d');

  var svg = d3.select("#scatter_plot_svg")
    .attr('width', fullWidthScatter)
    .attr('height', fullHeightScatter)
    .attr('class', 'chart')
    .append("g")
    .attr("transform", "translate(" + marginScatter.left + "," + marginScatter.top + ")");

  // draw the x axis
  var xAxis = d3.axisBottom(xScaleScatter)

  svg.append('g')
    .attr('transform', 'translate(0,' + heightScatter + ')')
    .attr('class', 'x-axis')
    .call(xAxis);

  svg.append("text")             
      .attr("transform",
            "translate(" + (widthScatter/2) + " ," + 
                           (heightScatter + marginScatter.top + 20) + ")")
      .style("text-anchor", "middle")
      .text("Prediction");

  // draw the y axis
  var yAxis = d3.axisLeft(yScaleScatter)

  svg.append('g')
    .attr('transform', 'translate(0,0)')
    .attr('class', 'y-axis ')
    .call(yAxis);

  svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - marginScatter.left)
      .attr("x",0 - (heightScatter / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Actual"); 

  //var g = main.append("svg:g").attr("id", "points");

  function drawPoint(x,y,value) {
      var cx = xScaleScatter(x);
      var cy = yScaleScatter(y);
      var r = 12
      //console.log(r)

      var vv = (parseFloat(value) / parseFloat(max) * 8);
      vv = parseInt(vv);

      console.log( x, y, vv, confusion_colormap[vv])

      context.fillStyle = confusion_colormap[vv];
      context.fillRect(cx-r,cy-r,r*2,r*2);

  }

  function draw() {
    context.clearRect(0, 0, widthScatter, heightScatter);
    data.forEach(function(point) {
      drawPoint(point['x'], point['y'], point['value']);
    });
  }

  draw();

}



