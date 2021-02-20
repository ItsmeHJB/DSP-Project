var nodes = []
var user_label_counter = 0;
var total_user_labels = 10;
var user_label_array = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"];

var our_ten_colours = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a'];
//var our_ten_colours = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd'];

var box_width = 100;
var box_height = 350;
var step = box_width + 5;

var unlabelled_box_x = 1;
var unlabelled_box_y = (box_height + 5);
var unlabelled_box_width = (box_width + 5) * 10 - 5;
var unlabelled_box_height = box_height / 4;

var rubbish_box_x = (box_width + 5) * 4;
var rubbish_box_y = (box_height + 5) * 2;
var rubbish_box_width = (box_width + 5);
var rubbish_box_height = box_height * 2;

var image_scale = 1.5;
var image_width = 28 * image_scale;
var image_height = 28 * image_scale;

var class_regions = []


var top_of_class_region = 56
var bottom_of_class_region = 350

// https://github.com/wbkd/d3-extended
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


function create_classifier_pane(placement) {

  var width = 1100,
      height = 450;

  // Create initial SVG background:
  var vis = d3.select(placement).append('svg');
  vis.attr('id','classifier_pane').attr('width', width).attr('height', height);

  // Generate classifier regions/boxes:
  var classes = 10;
  var regions = vis.append("g");
  for (i = 0; i < classes; i++) {
      regions.append('rect')
            .attr("id", function() { return "region_" + i; })
            .attr('x', (1 + (step * i)))
            .attr('y', 1)
            .attr('width', box_width)
            .attr('height', box_height)
            .style('fill', function () { return our_ten_colours[i] })
            .style("opacity", 0.75)
      regions.append('text')
            .attr('x', (10 + (step * i)))
            .attr('y', 40)
            .style("font-family", "sans-serif")
            .style("font-size", 16)
            .style('fill', "white")
            .style("opacity", 0.7)
            .text(user_label_array[i])

      var region_values = [(1 + (step * i)), 1, box_width+(1 + (step * i)), box_height+1, i]
      class_regions.push(region_values)

  }

  // Region for unclassified samples
  regions.append('rect')
        .attr("id", function() { return "region_?" })
        .attr('x', unlabelled_box_x)
        .attr('y', unlabelled_box_y)
        .attr('width', unlabelled_box_width)
        .attr('height', unlabelled_box_height)
        .style('fill', '#efefef');

}

function update_classifier_pane(images, batchsize) {

  console.log("update_classifier_pane");
  console.log(images, batchsize);

  user_label_counter = 0

  var width = 595
  var height = 440

  var nodes = []

  for (i = 0; i < batchsize; i++){
      nodes.push({ id: parseInt(images[0][i].split('_')[2].split('.')[0]),
        label: "?", 
        image: images[0][i], 
        scatter_id: parseInt(images[0][i].split('_')[2].split('.')[0]),
        x: unlabelled_box_x + (Math.random() * (unlabelled_box_width-image_width)),
        y: unlabelled_box_y + (Math.random() * (unlabelled_box_height-image_height)),
        moved: false
      })
  }

  console.log(nodes);

  var vis = d3.select('#classifier_pane');

  // Update the nodes…
  var node = vis.selectAll("g.node")
    .data(nodes, function(d) { return d.id; });

  var nodeEnter = node.enter().append("svg:g")
      .attr("class", "node")
      .on('mouseover', function(d) {
            //d3.select(this).moveToFront();
        });

  nodeEnter.append("svg:image")
      .attr("xlink:href",  function(nodes) {
          var src = "/static/images/cifar10_noclass/train/"
        return (src + nodes.image);
      })
      .attr("id", function(d) {return "image_" + d.scatter_id})
      .attr("label", function(d) {return d.label})
      .attr("x", function(d) { return d.x })
      .attr("y", function(d) { return d.y })
      .attr("height", image_height)
      .attr("width", image_width)
      .style("opacity", 1.0)
      .style("stroke", "red")
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)
      )
      .on('mouseover', function(d) {
            //d3.select(this).moveToFront();
        })

  nodeEnter.append("rect")
      .attr("id", function(d) {return "rect_" + d.scatter_id})
      .attr("x", function(d) { return d.x })
      .attr("y", function(d) { return d.y })
      .attr("height", image_height)
      .attr("width", image_width)
      .style("opacity", 0.4)
      .style("fill", "grey")
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)
      )
      .on('mouseover', function(d) {
            //d3.select(this).moveToFront();
        })

  node = vis.selectAll("g.node");

    function dragstarted(node) {

      node.x = d3.event.x;
      node.y = d3.event.y;

      node.moved = true;

      d3.select("#rect_" + node.scatter_id)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y });

      d3.select("#image_" + node.scatter_id)
        .attr("x", function() { return d3.event.x })
        .attr("y", function() { return d3.event.y });

      show_selected_node_in_scatter(node['scatter_id'], '?');
    }

    function dragged(node) {

      node.x = d3.event.x;
      node.y = d3.event.y;

      node.moved = true;

      d3.select("#rect_" + node.scatter_id)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y });

      d3.select("#image_" + node.scatter_id)
        .attr("x", function() { return d3.event.x })
        .attr("y", function() { return d3.event.y });

    }

    function dragended(node) {

      node.x = d3.event.x;
      node.y = d3.event.y;

      node.moved = true;

      d3.select("#rect_" + node.scatter_id)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y });

      d3.select("#image_" + node.scatter_id)
        .attr("x", function() { return d3.event.x })
        .attr("y", function() { return d3.event.y });

      id = node.id;
      label = node.label;
      image = node.image;
      xpos = node.x;
      ypos = node.y;
      scatter_id = node.scatter_id

      var this_label = '-1'
      for (i in class_regions) {
        var r = class_regions[i];
        //console.log(r)
        if (xpos > r[0] && xpos < r[2] && ypos > r[1] && ypos < r[3]) {
          this_label = r[4]
        }
      }

      console.log("Label is ", this_label)
      send_data = {"id" : id, 
                    "image" : image, 
                    "xpos" : xpos, 
                    "ypos" : ypos, 
                    "new_label" : this_label,
                    "user_confidence" : (Math.abs(ypos - bottom_of_class_region)) / (box_height - image_height),
                    "user_labelled" : 'true'
                  }

      console.log("Send_data", send_data);
      console.log("User confidence: ", send_data['user_confidence'])

        $.ajax({
          url : "update_label",
          data : send_data,
          success : function(data) { 

            all_data = JSON.parse(data);

            node.label = all_data['label']
            

            if (all_data['label'] != '?') {
              if (all_data['label'] != label) {

                d3.select("#rect_" + node.scatter_id)
                    .style("fill", "red");
                d3.select("#image_" + node.scatter_id)
                    .style("opacity", 0.5);

                user_label_counter += 1
                total_user_labels += 1
                console.log("Relabelling: ", label, " -> ", all_data['label']);
                console.log("Label has changed! Counter: ", user_label_counter);

                total_samples = all_data['total_samples']
                document.getElementById("sample_total-out").innerHTML = total_samples;
                document.getElementById("user_label_count-out").innerHTML = total_user_labels;
              } else {
                d3.select("#rect_" + node.scatter_id)
                    .style("fill", "blue");
                d3.select("#image_" + node.scatter_id)
                    .style("opacity", 0.5);
                total_samples = all_data['total_samples']
                document.getElementById("sample_total-out").innerHTML = total_samples;
                document.getElementById("user_label_count-out").innerHTML = total_user_labels;
              }
              show_selected_node_in_scatter(scatter_id, all_data['label']);
            } else {
              show_selected_node_in_scatter(scatter_id, '?');
              d3.select("#rect_" + node.scatter_id)
                    .style("fill", "yellow");
              d3.select("#image_" + node.scatter_id)
                    .style("opacity", 0.5);
              total_samples = all_data['total_samples']
                document.getElementById("sample_total-out").innerHTML = total_samples;
                document.getElementById("user_label_count-out").innerHTML = total_user_labels;
            }
          }
        });

// here we go through all images and get latest position
console.log("here we go through all images and get latest position")
      node = vis.selectAll("g.node");
      for (n in node) {
        console.log(n);
      }
      

    }
}

function update_classifier_pane_with_single_instance(images, batchsize, batch_total, node_data) {

  console.log("update_classifier_pane_with_single_instance");

  console.log(node_data)
  keys = Object.keys(node_data)

  user_label_counter = 0

  var width = 595
  var height = 440

  var nodes = []

  console.log(images)

  if (keys.length == 0) {
    nodes.push({ id: parseInt(images[0][0].split('_')[2].split('.')[0]), 
      label: "?", 
      image: images[0][0], 
      scatter_id: parseInt(images[0][0].split('_')[2].split('.')[0]),
      x: unlabelled_box_x + (Math.random() * (unlabelled_box_width-image_width)),
      y: unlabelled_box_y + (Math.random() * (unlabelled_box_height-image_height)),
      moved: false
      })
  } else {
    // we have data from the classifier - use it
    in_data = node_data['node_data'][0]

      var label = in_data['label'];
      console.log(label);
      var confidence = in_data['confidence'];
      var use_confidence_for_machine_position = true;

      var position_for_y = class_regions[label][1] + (Math.random() * (class_regions[label][3] - class_regions[label][1]));
      if (use_confidence_for_machine_position) {
        position_for_y = class_regions[label][1] + ( (1-confidence) * (class_regions[label][3] - class_regions[label][1]));
      }

      nodes.push({ id: in_data['id'], 
                  label: in_data['label'],
                  image: in_data['image'], 
                  x: class_regions[label][0] + (0.5 * box_width), // + (0.5 * (class_regions[label][2] - class_regions[label][0])) + image_width,
                  y: position_for_y,
                  scatter_id: in_data['id'], //parseInt(node_data[0][i]['image'].split('_')[2].split('.')[0]),
                  predictions: in_data['predictions'],
                  moved: false
                })

    
  

  }



  

  var vis = d3.select('#classifier_pane');

  // Update the nodes…
  var node = vis.selectAll("g.node")
    .data(nodes, function(d) { return d.id; });

  var nodeEnter = node.enter().append("svg:g")
      .attr("class", "node")
      .on('mouseover', function(d) {
            //d3.select(this).moveToFront();
        });

  nodeEnter.append("svg:image")
      .attr("xlink:href",  function(nodes) {
          var src = "/static/images/cifar10_noclass/train/"
        return (src + nodes.image);
      })
      .attr("id", function(d) {return "image_" + d.scatter_id})
      .attr("label", function(d) {return d.label})
      .attr("x", function(d) { return d.x })
      .attr("y", function(d) { return d.y })
      .attr("height", image_height)
      .attr("width", image_width)
      .style("opacity", 1.0)
      .style("stroke", "red")
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)
      )
      .on('mouseover', function(d) {
            //d3.select(this).moveToFront();
        })

  nodeEnter.append("rect")
      .attr("id", function(d) {return "rect_" + d.scatter_id})
      .attr("x", function(d) { return d.x })
      .attr("y", function(d) { return d.y })
      .attr("height", image_height)
      .attr("width", image_width)
      .style("opacity", 0.4)
      .style("fill", "grey")
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)
      )
      .on('mouseover', function(d) {
            //d3.select(this).moveToFront();
        })


  // Re-select for update.
  node = vis.selectAll("g.node");

    function dragstarted(node) {

      node.x = d3.event.x;
      node.y = d3.event.y;

      node.moved = true;

      d3.select("#rect_" + node.scatter_id)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y });

      d3.select("#image_" + node.scatter_id)
        .attr("x", function() { return d3.event.x })
        .attr("y", function() { return d3.event.y });

      show_selected_node_in_scatter(node['scatter_id'], '?');
    }

    function dragged(node) {

      node.x = d3.event.x;
      node.y = d3.event.y;

      node.moved = true;

      d3.select("#rect_" + node.scatter_id)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y });

      d3.select("#image_" + node.scatter_id)
        .attr("x", function() { return d3.event.x })
        .attr("y", function() { return d3.event.y });

    }

    function dragended(node) {

      node.x = d3.event.x;
      node.y = d3.event.y;

      node.moved = true;

      d3.select("#rect_" + node.scatter_id)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y });

      d3.select("#image_" + node.scatter_id)
        .attr("x", function() { return d3.event.x })
        .attr("y", function() { return d3.event.y });

      id = node.id;
      label = node.label;
      image = node.image;
      xpos = node.x;
      ypos = node.y;
      scatter_id = node.scatter_id

      var this_label = '-1'
      for (i in class_regions) {
        var r = class_regions[i];
        //console.log(r)
        if (xpos > r[0] && xpos < r[2] && ypos > r[1] && ypos < r[3]) {
          this_label = r[4]
        }
      }

      console.log("Label is ", this_label)
      send_data = {"id" : id, 
                    "image" : image, 
                    "xpos" : xpos, 
                    "ypos" : ypos, 
                    "new_label" : this_label,
                    "user_confidence" : (Math.abs(ypos - bottom_of_class_region)) / (box_height - image_height),
                    "user_labelled" : 'true'
                  }

      console.log("Send_data", send_data);
      console.log("User confidence: ", send_data['user_confidence'])

        $.ajax({
          url : "update_label",
          data : send_data,
          success : function(data) { 

            all_data = JSON.parse(data);

            node.label = all_data['label']
            

            if (all_data['label'] != '?') {
              if (all_data['label'] != label) {

                d3.select("#rect_" + node.scatter_id)
                    .style("fill", "red");
                d3.select("#image_" + node.scatter_id)
                    .style("opacity", 0.5);

                user_label_counter += 1
                total_user_labels += 1
                console.log("Relabelling: ", label, " -> ", all_data['label']);
                console.log("Label has changed! Counter: ", user_label_counter);

                total_samples = all_data['total_samples']
                document.getElementById("sample_total-out").innerHTML = total_samples;
                document.getElementById("user_label_count-out").innerHTML = total_user_labels;
              } else {
                d3.select("#rect_" + node.scatter_id)
                    .style("fill", "blue");
                d3.select("#image_" + node.scatter_id)
                    .style("opacity", 0.5);
                total_samples = all_data['total_samples']
                document.getElementById("sample_total-out").innerHTML = total_samples;
                document.getElementById("user_label_count-out").innerHTML = total_user_labels;
              }
              show_selected_node_in_scatter(scatter_id, all_data['label']);
            } else {
              show_selected_node_in_scatter(scatter_id, '?');
              d3.select("#rect_" + node.scatter_id)
                    .style("fill", "yellow");
              d3.select("#image_" + node.scatter_id)
                    .style("opacity", 0.5);
              total_samples = all_data['total_samples']
                document.getElementById("sample_total-out").innerHTML = total_samples;
                document.getElementById("user_label_count-out").innerHTML = total_user_labels;
            }
          }
        });


// here we go through all images and get latest position
console.log("here we go through all images and get latest position")
      node = vis.selectAll("g.node");
      for (n in node) {
        console.log(n);
      }

    }
}

function set_opacity_of_existing_images() {
  var vis = d3.select('#classifier_pane');
  vis.selectAll("g.node")
    .selectAll("image")
    .style("opacity", function(d) {return 0.1});

  vis.selectAll("g.node")
    .selectAll("rect")
    .style("opacity", function(d) {return 0.2});
    //.each( function () {
    //  d3.select(this)
    //      .on("mouseover")
    //})

    //vis.selectAll("g.node")
    //.selectAll("rect")
    //.attr("opacity", function(d) {return 0.5});
}

function destroy_images_with_no_label() {
  var vis = d3.select('#classifier_pane');
  vis.selectAll("g.node")
  .each( function () {
    console.log(d3.select(this))
      
          
    })
}

function update_classifier_pane_predict(node_data, batchsize) {
  console.log("update_classifier_pane_predict");
  console.log(node_data, batchsize)

  user_label_counter = 0

  var nodes = []

  var width = 595
  var height = 440

  for (i = 0; i < (batchsize); i++){

    //console.log(node_data[0][i])

    var label = node_data[0][i]['label'];
    console.log(label);

    var confidence = node_data[0][i]['confidence'];

    //console.log( "Class region:", class_regions )
    //console.log( "Class label region:", class_regions[label] )

    var use_confidence_for_machine_position = true;

    var position_for_y = class_regions[label][1] + (Math.random() * (class_regions[label][3] - class_regions[label][1]));
    if (use_confidence_for_machine_position) {
      position_for_y = class_regions[label][1] + ( (1-confidence) * (class_regions[label][3] - class_regions[label][1]));
    }

    nodes.push({ id: node_data[0][i]['id'], 
                label: node_data[0][i]['label'],
                image: node_data[0][i]['image'], 
                x: class_regions[label][0] + (0.5 * box_width), // + (0.5 * (class_regions[label][2] - class_regions[label][0])) + image_width,
                y: position_for_y,
                scatter_id: node_data[0][i]['id'], //parseInt(node_data[0][i]['image'].split('_')[2].split('.')[0]),
                predictions: node_data[0][i]['predictions'],
                moved: false
              })

    
  }

  //console.log(nodes)

  var vis = d3.select('#classifier_pane');

  // Update the nodes…
  var node = vis.selectAll("g.node")
    .data(nodes, function(d) { return d.id; });

  var nodeEnter = node.enter().append("svg:g")
      .attr("class", "node")
      .on('mouseover', function(d) {
            //d3.select(this).moveToFront();
        });

  nodeEnter.append("svg:image")
      .attr("xlink:href",  function(nodes) {
          var src = "/static/images/cifar10_noclass/train/"
        return (src + nodes.image);
      })
      .attr("id", function(d) {return "image_" + d.scatter_id})
      .attr("label", function(d) {return d.label})
      .attr("x", function(d) { return d.x })
      .attr("y", function(d) { return d.y })
      .attr("height", image_height)
      .attr("width", image_width)
      .style("opacity", 1.0)
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)
      )
      .on('mouseover', function(d) {
            //d3.select(this).moveToFront();
        })

  nodeEnter.append("rect")
      .attr("id", function(d) {return "rect_" + d.scatter_id})
      .attr("x", function(d) { return d.x })
      .attr("y", function(d) { return d.y })
      .attr("height", image_height)
      .attr("width", image_width)
      .style("opacity", 0.4)
      .style("fill", "yellow")
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)
      )
      .on('mouseover', function(d) {
            //d3.select(this).moveToFront();
        })

  // Re-select for update.
  node = vis.selectAll("g.node");

    function dragstarted(node) {

      node.x = d3.event.x;
      node.y = d3.event.y;

      node.moved = true;

      d3.select("#rect_" + node.scatter_id)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y });

      d3.select("#image_" + node.scatter_id)
        .attr("x", function() { return d3.event.x })
        .attr("y", function() { return d3.event.y });

      show_selected_node_in_scatter(node['scatter_id'], '?');
    }

    function dragged(node) {

      node.x = d3.event.x;
      node.y = d3.event.y;

      node.moved = true;

      d3.select("#rect_" + node.scatter_id)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y });

      d3.select("#image_" + node.scatter_id)
        .attr("x", function() { return d3.event.x })
        .attr("y", function() { return d3.event.y });

    }

    function dragended(node) {

      node.x = d3.event.x;
      node.y = d3.event.y;

      node.moved = true;

      d3.select("#rect_" + node.scatter_id)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y });

      d3.select("#image_" + node.scatter_id)
        .attr("x", function() { return d3.event.x })
        .attr("y", function() { return d3.event.y });

      id = node.id;
      label = node.label;
      image = node.image;
      xpos = node.x;
      ypos = node.y;
      scatter_id = node.scatter_id

      var this_label = '-1'
      for (i in class_regions) {
        var r = class_regions[i];
        //console.log(r)
        if (xpos > r[0] && xpos < r[2] && ypos > r[1] && ypos < r[3]) {
          this_label = r[4]
        }
      }

      console.log("Label is ", this_label)
      send_data = {"id" : id, 
                    "image" : image, 
                    "xpos" : xpos, 
                    "ypos" : ypos, 
                    "new_label" : this_label,
                    "user_confidence" : (Math.abs(ypos - bottom_of_class_region)) / (box_height - image_height),
                    "user_labelled" : 'true'
                  }

      console.log("Send_data", send_data);
      console.log("User confidence: ", send_data['user_confidence'])


        $.ajax({
          url : "update_label",
          data : send_data,
          success : function(data) { 

            all_data = JSON.parse(data);

            node.label = all_data['label']
            

            if (all_data['label'] != '?') {
              if (all_data['label'] != label) {

                d3.select("#rect_" + node.scatter_id)
                    .style("fill", "red");
                d3.select("#image_" + node.scatter_id)
                    .style("opacity", 0.5);

                user_label_counter += 1
                total_user_labels += 1
                console.log("Relabelling: ", label, " -> ", all_data['label']);
                console.log("Label has changed! Counter: ", user_label_counter);

                total_samples = all_data['total_samples']
                document.getElementById("sample_total-out").innerHTML = total_samples;
                document.getElementById("user_label_count-out").innerHTML = total_user_labels;
              } else {
                d3.select("#rect_" + node.scatter_id)
                    .style("fill", "blue");
                d3.select("#image_" + node.scatter_id)
                    .style("opacity", 0.5);

                total_samples = all_data['total_samples']
                document.getElementById("sample_total-out").innerHTML = total_samples;
                document.getElementById("user_label_count-out").innerHTML = total_user_labels;
              }
              show_selected_node_in_scatter(scatter_id, all_data['label']);
            } else {
              show_selected_node_in_scatter(scatter_id, '?');
              d3.select("#rect_" + node.scatter_id)
                    .style("fill", "yellow");
              d3.select("#image_" + node.scatter_id)
                    .style("opacity", 0.5);
              total_samples = all_data['total_samples']
                document.getElementById("sample_total-out").innerHTML = total_samples;
                document.getElementById("user_label_count-out").innerHTML = total_user_labels;
            }
          }
        });

// here we go through all images and get latest position
console.log("here we go through all images and get latest position")
      node = vis.selectAll("g.node");
      for (n in node) {
        console.log(n);
      }

      

    }
    
}
