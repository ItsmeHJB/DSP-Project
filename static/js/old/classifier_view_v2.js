var nodes = []
var user_label_counter = 0;
var user_label_array = [];

var our_ten_colours = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a'];
//var our_ten_colours = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd'];

var box_width = 100;
var box_height = 350;
var step = box_width + 5;

var unlabelled_box_x = 1;
var unlabelled_box_y = (box_height + 5);
var unlabelled_box_width = (box_width + 5) * 10 - 5;
var unlabelled_box_height = box_height;

var rubbish_box_x = (box_width + 5) * 4;
var rubbish_box_y = (box_height + 5) * 2;
var rubbish_box_width = (box_width + 5);
var rubbish_box_height = box_height * 2;

var image_scale = 2;
var image_width = 28 * image_scale;
var image_height = 28 * image_scale;

var class_regions = []


var top_of_class_region = 56
var bottom_of_class_region = 350


function create_classifier_pane(placement) {

  var width = 1200,
      height = 700;

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
            .attr('x', (30 + (step * i)))
            .attr('y', 85)
            .style("font-family", "sans-serif")
            .style("font-size", 114)
            .style('fill', "white")
            .style("opacity", 0.3)
            //.style('fill', function () { return our_ten_colours[i] })
            .text(i)

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

  var width = 595
  var height = 440

  var nodes = []

  for (i = 0; i < batchsize; i++){
      nodes.push({ id: i, label: "?", 
      image: images[0][i], 
      scatter_id: parseInt(images[0][i].split('_')[2].split('.')[0]) })
  }

  console.log(nodes);

  var vis = d3.select('#classifier_pane');

  // Update the nodes…
  var node = vis.selectAll("g.node")
    .data(nodes, function(d) { return d.id; });

  // Enter any new nodes.
  var nodeEnter = node.enter().append("svg:g")
      .attr("class", "node");

  var circle = nodeEnter.append("svg:circle")
      .attr("r", 20)
      .style("fill", "#F35");


  var image = nodeEnter.append("svg:image")
      .attr("xlink:href",  function(nodes) {
          var src = "/static/images/cifar10_noclass/train/"
        return (src + nodes.image);
      })
      .attr("x", function() { return unlabelled_box_x + (Math.random() * (unlabelled_box_width-image_width)) })
      .attr("y", function() { return unlabelled_box_y + (Math.random() * (unlabelled_box_height-image_height)) })
      .attr("height", image_height)
      .attr("width", image_width)
      .attr("opacity", 1.0)
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)
      );

  // Re-select for update.
  node = vis.selectAll("g.node");

    function dragstarted(node) {
      //console.log("dragstart");
      //console.log(node['scatter_id']);
      show_selected_node_in_scatter(node['scatter_id'], '?');
      node.x = d3.event.x; // - (image_width / 2);
      node.y = d3.event.y; // - (image_height / 2);
    }

    function dragged(node) {

      //console.log("dragged");
      //console.log(node['scatter_id'])

      node.x = d3.event.x; // - (image_width / 2);
      node.y = d3.event.y; // - (image_height / 2);

      d3.select(this)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y })

    }

    function dragended(node) {

      node.x = d3.event.x; // + (image_width / 2);
      node.y = d3.event.y; // + (image_height / 2);

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
          total_samples = all_data['total_samples']
          document.getElementById("sample_total-out").innerHTML = total_samples;

          if (all_data['label'] != '?') {
            if (all_data['label'] != label) {
              user_label_counter += 1
              console.log("Relabelling: ", label, " -> ", all_data['label']);
              console.log("Label has changed! Counter: ", user_label_counter);
            }
            show_selected_node_in_scatter(scatter_id, all_data['label']);
          } else {
            show_selected_node_in_scatter(scatter_id, '?');
          }
        }
      });
      



      
      // node.x = null;
      // node.y = null;
    }
}

function update_classifier_pane_with_single_instance(images, batchsize, batch_total) {

  console.log("update_classifier_pane");
  //console.log(images, batchsize, batch_total);

  var width = 595
  var height = 440

  var nodes = []

  for (i = 0; i < batchsize; i++){
    nodes.push({ id: i + nodes.length, 
      label: "?", 
      image: images[0][i], 
      scatter_id: parseInt(images[0][i].split('_')[2].split('.')[0]) })
  }

  console.log(nodes);

  var vis = d3.select('#classifier_pane');

  // Update the nodes…
  var node = vis.selectAll("g.node")
    .data(nodes, function(d) { return d.id; });

  // Enter any new nodes.
  var nodeEnter = node.enter().append("svg:g")
      .attr("class", "node");


  var image = nodeEnter.append("svg:image")
      .attr("xlink:href",  function(nodes) {
          var src = "/static/images/cifar10_noclass/train/"
        return (src + nodes.image);
      })
      .attr("x", function() { return unlabelled_box_x + (Math.random() * (unlabelled_box_width-image_width)) })
      .attr("y", function() { return unlabelled_box_y + (Math.random() * (unlabelled_box_height-image_height)) })
      .attr("height", image_height)
      .attr("width", image_width)
      .attr("opacity", 1.0)
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)
      );


  // Re-select for update.
  node = vis.selectAll("g.node");

    function dragstarted(node) {
      show_selected_node_in_scatter(node['scatter_id'], '?');
      node.x = d3.event.x; // - (image_width / 2);
      node.y = d3.event.y; // - (image_height / 2);
    }

    function dragged(node) {
      node.x = d3.event.x; // - (image_width / 2);
      node.y = d3.event.y; // - (image_height / 2);

      d3.select(this)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y })

    }

    function dragended(node) {

      node.x = d3.event.x; // + (image_width / 2);
      node.y = d3.event.y; // + (image_height / 2);

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
          total_samples = all_data['total_samples']
          document.getElementById("sample_total-out").innerHTML = total_samples;

          if (all_data['label'] != '?') {
            if (all_data['label'] != label) {
              user_label_counter += 1
              console.log("Relabelling: ", label, " -> ", all_data['label']);
              console.log("Label has changed! Counter: ", user_label_counter);
            }
            show_selected_node_in_scatter(scatter_id, all_data['label']);
          } else {
            show_selected_node_in_scatter(scatter_id, '?');
          }
        }
      });

    }
}

function set_opacity_of_existing_images() {
  var vis = d3.select('#classifier_pane');

  // Update the nodes…
  var node = vis.selectAll("g.node").attr("opacity", 0.25).attr("fixed", true);
  var node = vis.selectAll("g.nodeuser").attr("opacity", 0.25 ).attr("fixed", true);

  // can we get the opacity value and decrease this in stages rather than as a fixed value
    
}

function update_classifier_pane_predict(node_data, batchsize) {
  console.log("update_classifier_pane_predict");
  console.log(node_data, batchsize)

  var nodes = []

  var width = 595
  var height = 440

  for (i = 0; i < (batchsize); i++){

    console.log(node_data[0][i])

    var label = node_data[0][i]['label'];
    console.log(label);

    var confidence = node_data[0][i]['confidence'];

    console.log( "Class region:", class_regions )
    console.log( "Class label region:", class_regions[label] )

    var use_confidence_for_machine_position = true;
    var position_for_y = class_regions[label][1] + (Math.random() * (class_regions[label][3] - class_regions[label][1]));
    if (use_confidence_for_machine_position) {
      position_for_y = class_regions[label][1] + ( (1-confidence) * (class_regions[label][3] - class_regions[label][1]));
    }

    nodes.push({ id: node_data[0][i]['id'], 
                label: node_data[0][i]['label'],
                image: node_data[0][i]['image'], 
                x: class_regions[label][0] + (0.5 * (class_regions[label][2] - class_regions[label][0])) + image_width,
                y: position_for_y,
                scatter_id: parseInt(node_data[0][i]['image'].split('_')[2].split('.')[0]),
                predictions: node_data[0][i]['predictions']
              })

    
  }

  //console.log(nodes)

  var vis = d3.select('#classifier_pane');

  // Update the nodes…
  var node = vis.selectAll("g.node")
    .data(nodes, function(d) { return d.id; });
    //.attr("opacity", 0.4);

  // Update the nodes…
  //var node = vis.selectAll("g.nodeuser")
    //.data(nodes, function(d) { return d.id; })
    //.attr("opacity", 0.4);

  // Enter any new nodes.
  var nodeEnter = node.enter().append("svg:g")
      .attr("class", "nodeuser");

  // Append images:
  var image = nodeEnter.append("svg:image")
      .attr("xlink:href",  function(nodes) {
        var src = "/static/images/cifar10_noclass/train/"
        return (src + nodes.image);}
      )
      .attr("class", "nodeuser")
      .attr('x', function (node) { return node.x })
      .attr('y', function (node) { return node.y })
      .attr("height", image_height)
      .attr("width", image_width)
      .attr("opacity", 1.0)
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)
      );

  // Re-select for update.
  node = vis.selectAll("g.node");

    function dragstarted(node) {
      console.log("Node", node)
      console.log("dragstarted");
      show_selected_node_in_scatter(node['scatter_id'], '?'); // node['label']);
      node.x = d3.event.x; // - (image_width / 2);
      node.y = d3.event.y; // - (image_height / 2);
    }

    function dragged(node) {

      node.x = d3.event.x; // - (image_width / 2);
      node.y = d3.event.y; // - (image_height / 2);

      d3.select(this)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y })
    }

    function dragended(node) {

      node.x = d3.event.x; // + (image_width / 2);
      node.y = d3.event.y; // + (image_height / 2);

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
          total_samples = all_data['total_samples']
          document.getElementById("sample_total-out").innerHTML = total_samples;

          if (all_data['label'] != '?') {
            if (all_data['label'] != label) {
              user_label_counter += 1
              console.log("Relabelling: ", label, " -> ", all_data['label']);
              console.log("Label has changed! Counter: ", user_label_counter);
            }
            show_selected_node_in_scatter(scatter_id, all_data['label']);
          } else {
            show_selected_node_in_scatter(scatter_id, '?');
          }
        }
      });

    }
    
}
