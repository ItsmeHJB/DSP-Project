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



var marginConfusion = { top: 25, right: 50, bottom: 50, left: 30 },
    fullWidthConfusion = 425,
    fullHeightConfusion = 380,
    widthConfusion = fullWidthConfusion - marginConfusion.left - marginConfusion.right,
    heightConfusion = fullHeightConfusion - marginConfusion.top - marginConfusion.bottom;

var confusion_colormap = ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d']

var xScaleConfusion = d3.scaleLinear()
  .domain([-1 , 10])
  .range([0 , widthConfusion]);

var yScaleConfusion = d3.scaleLinear()
  .domain([-1 , 10])
  .range([heightConfusion, 0]);

var confusion_data;

function create_confusion_plot(placement, data) {

  var numrows = 10;
  var numcols = 10;

  var max = 0

  for (i in data) {
    if (max < data[i]['value']) {
      max = data[i]['value'];
    }
  }

  confusion_data = data


  console.log(confusion_data);


  var canvas = d3.select("#confusion_plot_canvas")
    .attr('width', widthConfusion - 1)
    .attr('height', heightConfusion - 1)
    .style("transform", "translate(" + (marginConfusion.left - fullWidthConfusion ) +
            "px" + "," + (marginConfusion.top + 1) + "px" + ")");

  var context = canvas.node().getContext('2d');

  d3.select("#confusion_plot_svg").selectAll("g").remove();

  var svg = d3.select("#confusion_plot_svg")
    .attr('width', fullWidthConfusion)
    .attr('height', fullHeightConfusion)
    .attr('class', 'chart')
    .append("g")
    .attr("transform", "translate(" + marginConfusion.left + "," + marginConfusion.top + ")");

  // draw the x axis
  var xAxis = d3.axisBottom(xScaleConfusion)

  svg.append('g')
    .attr('transform', 'translate(0,' + heightConfusion + ')')
    .attr('class', 'x-axis')
    .call(xAxis);

  svg.append("text")             
      .attr("transform",
            "translate(" + (widthConfusion/2) + " ," + 
                           (heightConfusion + marginConfusion.top + 20) + ")")
      .style("text-anchor", "middle")
      .text("Prediction");

  // draw the y axis
  var yAxis = d3.axisLeft(yScaleConfusion)

  svg.append('g')
    .attr('transform', 'translate(0,0)')
    .attr('class', 'y-axis ')
    .call(yAxis);

  svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - marginConfusion.left)
      .attr("x",0 - (heightConfusion / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Actual"); 

  //var g = main.append("svg:g").attr("id", "points");

  function drawPoint(x,y,value) {
      var cx = xScaleConfusion(x);
      var cy = yScaleConfusion(y);
      var r = 12
      //console.log(r)
      var vv = 0
      if (max > 0) {
        vv = (parseFloat(value) / parseFloat(max) * 8);
        vv = parseInt(vv);
      }

      

      console.log( x, y, vv, confusion_colormap[vv])

      context.fillStyle = confusion_colormap[vv];
      context.fillRect(cx-r,cy-r,r*2,r*2);

  }

  function draw() {
    context.clearRect(0, 0, widthConfusion, heightConfusion);
    data.forEach(function(point) {
      drawPoint(point['x'], point['y'], point['value']);
    });
  }

  draw();

}



