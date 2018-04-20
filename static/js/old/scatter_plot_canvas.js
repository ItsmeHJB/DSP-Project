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

/*
var xScaleScatter = d3.scaleLinear()
  .domain([-20 , +20])
  .range([0 , widthScatter]);

var yScaleScatter = d3.scaleLinear()
  .domain([-20 , +20])
  .range([heightScatter, 0]);
*/

var xScaleScatter = d3.scaleLinear()
  .domain([-1 , +1])
  .range([0 , widthScatter]);

var yScaleScatter = d3.scaleLinear()
  .domain([-1 , +1])
  .range([heightScatter, 0]);

var scatter_data;

function create_scatter_plot(data, placement) {

  scatter_data = data;

  var minX = d3.min(data, function(d) {
      return d[0];
    });

  var maxX = d3.max(data, function(d) {
      return d[0];
    });

  var minY = d3.min(data, function(d) {
      return d[1];
    });

  var maxY = d3.max(data, function(d) {
      return d[1];
    });

  console.log(minX, maxX, minY, maxX)

  data.forEach(function(d, i) {
    //console.log(d);
    d[0] = ((d[0] - minX) / (maxX - minX)) * 2 - 1;
    d[1] = ((d[1] - minY) / (maxY - minY)) * 2 - 1;
    d[6] = i;
  });

  console.log(data);


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

  // draw the y axis
  var yAxis = d3.axisLeft(yScaleScatter)

  svg.append('g')
    .attr('transform', 'translate(0,0)')
    .attr('class', 'y-axis ')
    .call(yAxis);

  //var g = main.append("svg:g").attr("id", "points");

  

  function drawPoint(x,y) {
      var cx = xScaleScatter(x);
      var cy = yScaleScatter(y);
      var r = 2;
      context.beginPath();
      context.arc(cx, cy, r, 0, 2 * Math.PI);
      context.closePath();
      context.fill();
  }

  function draw(index) {
    context.clearRect(0, 0, widthScatter, heightScatter);
    context.fillStyle = 'rgba(200,200,200,0.2)';
    data.forEach(function(point) {
      drawPoint(point[0], point[1]);
    });

    canvas.on("mousemove",function() {
      var xy = d3.mouse(this);
      //console.log(xy);

      var cx = xScaleScatter.invert(xy[0]);
      var cy = yScaleScatter.invert(xy[1]);
      console.log(cx, cy);

      var color = context.getImageData(xy[0], xy[1], 1, 1).data;
      if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
        // a blank region of chart
      } else {
        $.ajax({
          url:'get_image_from_scatter_position',
          data:{'cx': cx, 'cy': cy},
          success: function(msg) {
            msg = JSON.parse(msg);
            console.log(msg);
          }
        })
      }


    });

  }

  draw();

}

function create_scatter_plot_with_all_labels(data, placement, label_value) {

  // label_value can be 2 for actual, 3 for predicted, or 4 for user
  if (label_value < 2 && label_value > 4) {
    label_value = 2;
    // default
  }

  console.log('create_scatter_plot_with_all_labels');

  var minX = d3.min(data, function(d) {
      return d[0];
    });

  var maxX = d3.max(data, function(d) {
      return d[0];
    });

  var minY = d3.min(data, function(d) {
      return d[1];
    });

  var maxY = d3.max(data, function(d) {
      return d[1];
    });

  console.log(minX, maxX, minY, maxX)

  data.forEach(function(d) {
    //console.log(d);
    d[0] = ((d[0] - minX) / (maxX - minX)) * 2 - 1;
    d[1] = ((d[1] - minY) / (maxY - minY)) * 2 - 1;
  });

  console.log(data);


    //data[d][0] = ((data[d][0] - minX) / (maxX - minX)) * 2 - 1;
    //data[d][1] = ((data[d][1] - minY) / (maxY - minY)) * 2 - 1;
  //}


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

  // draw the y axis
  var yAxis = d3.axisLeft(yScaleScatter)

  svg.append('g')
    .attr('transform', 'translate(0,0)')
    .attr('class', 'y-axis ')
    .call(yAxis);

  //var g = main.append("svg:g").attr("id", "points");

  

  function drawPoint(x,y) {
      var cx = xScaleScatter(x);
      var cy = yScaleScatter(y);
      var r = 1;
      context.beginPath();
      context.arc(cx, cy, r, 0, 2 * Math.PI);
      context.closePath();
      context.fill();
  }

  function draw(index) {
    context.clearRect(0, 0, widthScatter, heightScatter);
    
    context.fillStyle = 'rgba(200,200,200,0.2)';

    data.forEach(function(point) {
      //console.log(point)
      context.fillStyle = our_ten_colours[point[label_value]];
      drawPoint(point[0], point[1]);
    });
  }

  draw();

}




/*
  g.selectAll(".dot")
    .data(data)
    .enter().append("circle")
    .attr("id", function(d,i) { return "circle_" + i; })
    .attr("cx", function(d, i) {
      return x(d[0])
    })
    .attr("cy", function(d) {
      return y(d[1])
    })
    .attr("r", 2)
    .style("opacity", 0.1)
    //.style("stroke", "black")
    .style("fill", "grey");
    //.on("mouseover", function(d,i) { d3.select(this).moveToFront(); console.log(d,i); d3.select(this).style("stroke","orange"); })
    //.on("mouseout", function(d,i) { console.log(d,i); d3.select(this).style("stroke","grey"); })
*/


function highlight_point_in_scatterplot(image_id, label) {
  console.log("does this work?", image_id)
  d3.select("#scatter_plot-view")
    .select("svg")
    .select("#circle_" + image_id)
    .style("opacity", 1)
    .style("fill", function() {return our_ten_colours[label] })
        .moveToFront();
}

function highlight_point_in_scatterplot_canvas(node) {
  console.log('highlight_point_in_scatterplot_canvas')
  var canvas = d3.select("#scatter_plot_canvas");
  var context = canvas.node().getContext('2d');

  this_class = node['label']
  this_x = node['scatter_x']
  this_y = node['scatter_y']

  function drawPoint(x,y) {
      var cx = xScaleScatter(x);
      var cy = yScaleScatter(y);
      var r = 3;
      context.beginPath();
      context.arc(cx, cy, r, 0, 2 * Math.PI);
      context.closePath();
      context.fill();
  }

  context.fillStyle = our_ten_colours[this_class];
  drawPoint(this_x,this_y)
}

function show_selected_node_in_scatter(node, label) {
  console.log('show_selected_node_in_scatter', node, label)
  var canvas = d3.select("#scatter_plot_canvas");
  var context = canvas.node().getContext('2d');

  console.log(scatter_data[node]);

  this_x = scatter_data[node][0]
  this_y = scatter_data[node][1]

  function drawPoint(x,y) {
      var cx = xScaleScatter(x);
      var cy = yScaleScatter(y);
      var r = 5;
      context.beginPath();
      context.arc(cx, cy, r, 0, 2 * Math.PI);
      context.closePath();
      context.fill();
  }

  if (label == '?') {
    context.fillStyle = 'black';
    scatter_data[node][4] = -1;
    scatter_data[node][5] = scatter_data[node][5] + 1;
  } else {
    context.fillStyle = our_ten_colours[label];
    scatter_data[node][4] = parseInt(label)
    scatter_data[node][5] = scatter_data[node][5] + 1;
  }
  
  drawPoint(this_x,this_y);

  console.log(scatter_data[node]);

}

function colour_scatter_based_on_prediction() {
  console.log("colour_scatter_based_on_prediction")
var canvas = d3.select("#scatter_plot_canvas");
  var context = canvas.node().getContext('2d');

  function drawPoint(x,y, user_point) {
      var cx = xScaleScatter(x);
      var cy = yScaleScatter(y);
      var r;
      if (user_point > -1) {
        r = 5;
      } else {
        r = 1;
      }
      context.beginPath();
      context.arc(cx, cy, r, 0, 2 * Math.PI);
      context.closePath();
      context.fill();

      if (user_point > -1) {
        context.strokeStyle = "#000000";
        context.lineWidth = 30;
        context.beginPath();
        context.arc(cx, cy, r, 0, 2 * Math.PI);
        context.stroke();
        //context.closePath();

        
        
      }
  }

  function draw_again(index) {
    context.clearRect(0, 0, widthScatter, heightScatter);
    
    scatter_data.forEach(function(point) {
      //console.log(point)
      context.fillStyle = our_ten_colours[point[3]];
      drawPoint(point[0], point[1], point[4]);
    });

    scatter_data.forEach(function(point) {
      if (point[4] > -1) {
        //console.log(point)


        context.fillStyle = our_ten_colours[point[3]];

        drawPoint(point[0], point[1], point[4]);
      }
      
    });

  }

  draw_again();

}

