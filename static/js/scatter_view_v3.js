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

  var minX = d3.min(scatter_data, function(d) {
      return d[0];
    });

  var maxX = d3.max(scatter_data, function(d) {
      return d[0];
    });

  var minY = d3.min(scatter_data, function(d) {
      return d[1];
    });

  var maxY = d3.max(scatter_data, function(d) {
      return d[1];
    });

  console.log(minX, maxX, minY, maxX)

  scatter_data.forEach(function(d, i) {
    //console.log(d);
    d[0] = ((d[0] - minX) / (maxX - minX)) * 2 - 1;
    d[1] = ((d[1] - minY) / (maxY - minY)) * 2 - 1;
    d[6] = i;
  });

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

  // draw the y axis
  var yAxis = d3.axisLeft(yScaleScatter)

  svg.append('g')
    .attr('transform', 'translate(0,0)')
    .attr('class', 'y-axis ')
    .call(yAxis);

  //var g = main.append("svg:g").attr("id", "points");

  function drawPoint(x,y, user_point, hover_point) {
      var cx = xScaleScatter(x);
      var cy = yScaleScatter(y);
      var r;

      if (user_point > -1) {
        r = 5;
      } else {
        r = 1;
        context.fillStyle = '#DDDDDD' + '88';
      }

      if (hover_point > -1) {
        // hightlight in some way
        context.fillStyle = '#FF0000' + '88';
      } else {
        // ignore
      }

      context.beginPath();
      context.arc(cx, cy, r, 0, 2 * Math.PI);
      context.closePath();
      context.fill();

      if (user_point > -1) {
        //console.log("User point > -1")
        context.beginPath();
        context.arc(cx, cy, r, 0, 2 * Math.PI);
        context.closePath();
        context.strokeStyle= "#000000" + '44';
        context.lineWidth = 2;
        context.stroke();
      }

  }

  function draw() {
    context.clearRect(0, 0, widthScatter, heightScatter);
    data.forEach(function(point) {
      drawPoint(point[0], point[1], -1, -1);
    });
  }



  canvas.on("mousemove",function() {
    var xy = d3.mouse(this);
    //console.log(xy);

    var cx = xScaleScatter.invert(xy[0]);
    var cy = yScaleScatter.invert(xy[1]);
    

    var color = context.getImageData(xy[0], xy[1], 1, 1).data;
    if (color[0] == 0 && color[1] == 0 && color[2] == 0) {

    } else {
      //console.log("Point on scatter:", cx, cy);
      my_distance = 100;
      my_index = -1;
      
      scatter_data.forEach(function(d, i) {
        //drawPoint(d[0], d[1], -1, -1);
        distance = Math.sqrt( ((d[0] - cx) * (d[0] - cx)) + ((d[1] - cy) * (d[1] - cy)) )
        if (distance < my_distance) {
          my_distance = distance;
          my_index = i;
        }
      });
      //console.log("Distance:", my_distance, "Index:", my_index);
      
      //drawPoint(scatter_data[my_index][0], scatter_data[my_index][1], -1, 1);

      var img = document.createElement("img");
      img.src = "/static/images/cifar10_noclass/train/CIFAR10_image_" + my_index + ".jpg"
      img.width = 32;
      img.height = 32;
      var dv = document.getElementById("scatter_plot-image_view");
      if (dv.hasChildNodes()) {
        dv.replaceChild(img, dv.lastChild);
      } else {
        dv.appendChild(img);
      }
      //while (dv.hasChildNodes()) { 
     //   dv.removeChild(dv.lastChild); 
      //} 
      

      /*
      $.ajax({
        url:'get_image_from_scatter_position',
        data:{'cx': cx, 'cy': cy},
        success: function(msg) {
          msg = JSON.parse(msg);
          console.log(msg);
        }
      })
*/
      
    }
  });

  canvas.on("click",function() {
    var xy = d3.mouse(this);
    //console.log(xy);

    var cx = xScaleScatter.invert(xy[0]);
    var cy = yScaleScatter.invert(xy[1]);
    

    var color = context.getImageData(xy[0], xy[1], 1, 1).data;
    if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
      // a blank region of chart
      //draw();
    } else {
      console.log("Point on scatter:", cx, cy);
      my_distance = 100;
      my_index = -1;
      //context.clearRect(0, 0, widthScatter, heightScatter);
      //context.fillStyle = '#DDDDDD' + '88';
      scatter_data.forEach(function(d, i) {
        //drawPoint(d[0], d[1], -1, -1);
        distance = Math.sqrt( ((d[0] - cx) * (d[0] - cx)) + ((d[1] - cy) * (d[1] - cy)) )
        if (distance < my_distance) {
          my_distance = distance;
          my_index = i;
        }
      });
      //console.log("Distance:", my_distance, "Index:", my_index);
      
      //drawPoint(scatter_data[my_index][0], scatter_data[my_index][1], -1, 1);

      var img = document.createElement("img");
      img.src = "/static/images/cifar10_noclass/train/CIFAR10_image_" + my_index + ".jpg"
      img.width = 64;
      img.height = 64;
      var dv = document.getElementById("scatter_plot-image_view");
      if (dv.hasChildNodes()) {
        dv.replaceChild(img, dv.lastChild);
      } else {
        dv.appendChild(img);
      }
      //while (dv.hasChildNodes()) { 
     //   dv.removeChild(dv.lastChild); 
      //} 
      

      var image_file_ref = [["CIFAR10_image_" + my_index + ".jpg"]]
      console.log("update with ", image_file_ref)
      

      
      $.ajax({
        url:'add_image_to_labelling_pool',
        data:{'my_index': my_index},
        success: function(data) {
          node_data = JSON.parse(data);
          update_classifier_pane_with_single_instance(image_file_ref, 1, 1, node_data);
        }
      })

    }
  } );



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
    
    context.fillStyle = '#DDDDDD' + '88';

    data.forEach(function(point) {
      //console.log(point)
      context.fillStyle = our_ten_colours[point[label_value]] + '88';
      drawPoint(point[0], point[1]);
    });
  }

  draw();

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
    context.fillStyle = "#000000" + '44';
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
        console.log("User point > -1")
        context.beginPath();
        context.arc(cx, cy, r, 0, 2 * Math.PI);
        context.closePath();
        context.strokeStyle= "#000000" + '44';
        context.lineWidth = 2;
        context.stroke();
      }

  }

  function draw_again(index) {
    context.clearRect(0, 0, widthScatter, heightScatter);
    
    scatter_data.forEach(function(point) {
      //console.log(point)
      context.fillStyle = our_ten_colours[point[3]] + '44';
      drawPoint(point[0], point[1], point[4]);
    });

    scatter_data.forEach(function(point) {
      if (point[4] > -1) {
        //console.log(point)
        context.fillStyle = our_ten_colours[point[3]]  + '88';
        drawPoint(point[0], point[1], point[4]);
      }
      
    });

  }

  draw_again();

}

