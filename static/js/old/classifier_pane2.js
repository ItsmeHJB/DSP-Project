var nodes = []
var user_label_counter = 0;
var user_label_array = [];

var our_ten_colours = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a'];
//var our_ten_colours = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd'];

var box_width = 200;
var box_height = 200;
var step = box_width + 5;

var unlabelled_box_x = 1;
var unlabelled_box_y = (box_height + 5) * 2;
var unlabelled_box_width = (box_width + 5) * 5 - 5;
var unlabelled_box_height = box_height * 2;

var rubbish_box_x = (box_width + 5) * 4;
var rubbish_box_y = (box_height + 5) * 2;
var rubbish_box_width = (box_width + 5);
var rubbish_box_height = box_height * 2;

var image_scale = 2;
var image_width = 28 * image_scale;
var image_height = 28 * image_scale;

function create_classifier_pane(placement) {

  var width = 1200,
      height = 800;

  // Create initial SVG background:
  var vis = d3.select(placement).append('svg');
  vis.attr('id','classifier_pane').attr('width', width).attr('height', height);


  

  // Generate classifier regions/boxes:
  var classes = 10;
  var regions = vis.append("g");
  for (i = 0; i < classes; i++) {
    
    
    if (i >= 5) { // bottom regions
      var j = (i-5) // Used with step below.
      regions.append('rect')
            .attr("id", function() { return "region_" + i; })
            .attr('x', (1 + (step * j)))
            .attr('y', box_height + 5)
            .attr('width', box_width)
            .attr('height', box_height)
            .style('fill', function () { return our_ten_colours[i] })
      regions.append('text')
            .attr('x', (30 + (step * j)))
            .attr('y', box_height + 60)
            .style("font-family", "sans-serif")
            .style("font-size", 40)
            .style("fill", "white")
            .text(i)
    } else { //top regions
      regions.append('rect')
            .attr("id", function() { return "region_" + i; })
            .attr('x', (1 + (step * i)))
            .attr('y', 1)
            .attr('width', box_width)
            .attr('height', box_height)
            .style('fill', function () { return our_ten_colours[i] })
      regions.append('text')
            .attr('x', (30 + (step * i)))
            .attr('y', 60)
            .style("font-family", "sans-serif")
            .style("font-size", 40)
            .style("fill", "white")
            .text(i)
    }
  }

  // Region for unclassified samples
  regions.append('rect')
        .attr("id", function() { return "region_?" })
        .attr('x', unlabelled_box_x)
        .attr('y', unlabelled_box_y)
        .attr('width', unlabelled_box_width)
        .attr('height', unlabelled_box_height)
        .style('fill', '#efefef');
/*
// THIS WAS TO SHOW A 'RUBBISH' AREA NEXT TO UNLABELLED - BUT I THINK WE CAN REMOVE THIS
// IF LEFT IN UNLABELED IT SHOULD BE ASSUMED TO BE RUBBISH / UNCONFIDENT
  regions.append('rect')
        .attr("id", function() { return "region_X" })
        .attr('x', rubbish_box_x)
        .attr('y', rubbish_box_y)
        .attr('width', rubbish_box_width)
        .attr('height', rubbish_box_height)
        .style('fill', '#ababab');
*/
  //update(images, batchsize);
}

function update_classifier_pane(images, batchsize, batch_total) {

  console.log("update_classifier_pane");
  //console.log(images, batchsize, batch_total);

  var width = 595
  var height = 440

  for (i = 0; i < batchsize; i++){
    nodes.push({ id: i + batch_total, label: "?", image: images[0][i], scatter_id: parseInt(images[0][i].split('_')[2].split('.')[0]) })
  }

  console.log(nodes);

  var vis = d3.select('#classifier_pane');

  // Update the nodes…
  var node = vis.selectAll("g.node")
    .data(nodes, function(d) { return d.id; });

  // Enter any new nodes.
  var nodeEnter = node.enter().append("svg:g")
      .attr("class", "node")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended)
      );


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

  // Exit any old nodes.
  node.exit().remove();

  // Re-select for update.
  node = vis.selectAll("g.node");

    function dragstarted(node) {
      //console.log("dragstart");
      //console.log(node['scatter_id']);
      show_selected_node_in_scatter(node['scatter_id'], '?');
      node.x = node.x;
      node.y = node.y;
    }

    function dragged(node) {

      //console.log("dragged");
      //console.log(node['scatter_id'])

      node.x = d3.event.x - (image_width / 2);
      node.y = d3.event.y - (image_height / 2);

      d3.select(this)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y })

    }

    function dragended(node) {

      //console.log("dragended");
      //console.log(node['scatter_id'])

      id = node.id;
      label = node.label;
      image = node.image;
      xpos = node.x;
      ypos = node.y;
      scatter_id = node.scatter_id

      console.log("Current Label: ", label);
      send_data = {"id" : id, "image" : image, "xpos" : xpos, "ypos" : ypos}
      console.log(send_data)

/*
 The ajax call is used to produce a label... 
*/

$.ajax({
        url : "produce_label",
        data : send_data,
        success : function(data) {
          all_data = JSON.parse(data);
          console.log("All data returned", all_data)
          // Update node with assigned label
          total_samples = all_data['total_samples']
          document.getElementById("sample_total-out").innerHTML = total_samples;

          if (all_data['label'] != '?') {
            
            nodes[id]['label'] = all_data['label']
            nodes[id]['scatter_x'] = all_data['scatter_x']
            nodes[id]['scatter_y'] = all_data['scatter_y']
            //document.getElementById("class-out").innerHTML = "Classification: " + all_data['label'];
            //console.log(node)

            if (nodes[id]['label'] != label) {
              user_label_counter += 1
              console.log("Relabelling: ", label, " -> ", nodes[id]['label']);
              console.log("Label has changed! Counter: ", user_label_counter);
            }

            show_selected_node_in_scatter(scatter_id, nodes[id]['label']);
          } else {
            //document.getElementById("class-out").innerHTML = "Classification: ?";
            show_selected_node_in_scatter(scatter_id, '?');
          }


          

        }
      });
      



      
      // node.x = null;
      // node.y = null;
    }
}

function set_opacity_of_existing_images() {
  var vis = d3.select('#classifier_pane');

  // Update the nodes…
  var node = vis.selectAll("g.node").attr("opacity", 0.4);
    //.data(nodes, function(d) { return d.id; })
    
}

function update_classifier_pane_predict(node_data, batchsize) {
  // console.log("Batch pulled so far: ", batchtotal)
  // console.log("Length of array passed: ", images[0].length)
  // var nodes = []

  console.log("update_classifier_pane_predict");
  console.log(node_data, batchsize)



  var width = 595
  var height = 440

  for (i = 0; i < (batchsize); i++){
    nodes.push({ id: node_data[0][i]['id'], 
                label: node_data[0][i]['label'],
                image: node_data[0][i]['image'], 
                x: node_data[0][i]['x'], 
                y: node_data[0][i]['y'],
                scatter_id: parseInt(node_data[0][i]['image'].split('_')[2].split('.')[0]),
                predictions: node_data[0][i]['predictions']
              })
  }

  //console.log(nodes)

  var vis = d3.select('#classifier_pane');

  // Update the nodes…
  var node = vis.selectAll("g.node")
    .data(nodes, function(d) { return d.id; })
    .attr("opacity", 0.4);

  // Enter any new nodes.
  var nodeEnter = node.enter().append("svg:g")
      .attr("class", "node")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended)
      );

  // Append images:
  var image = nodeEnter.append("svg:image")
      .attr("xlink:href",  function(nodes) {
        var src = "/static/images/cifar10_noclass/train/"
        return (src + nodes.image);})
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

  // Exit any old nodes.
  node.exit().remove();

  // Re-select for update.
  node = vis.selectAll("g.node");

    function dragstarted(node) {

      console.log("Node", node)

      console.log("dragstarted");
      show_selected_node_in_scatter(node['scatter_id'], '?'); // node['label']);


      node.x = node.x;
      node.y = node.y;
    }

    function dragged(node) {

      node.x = d3.event.x;// - (image_width / 2);
      node.y = d3.event.y;// - (image_height / 2);

      d3.select(this)
        .attr('x', function (node) { return node.x })
        .attr('y', function (node) { return node.y })
    }

    function dragended(node) {
      console.log("dragended");


      

      id = node.id;
      label = node.label;
      image = node.image;
      xpos = node.x;
      ypos = node.y;
      predictions = node.predictions;

      console.log(node);
      console.log(id, label)
      console.log(predictions)

      var max_pred = Math.max(...predictions);
      var prediction_label = predictions.indexOf(max_pred) ;

      console.log(max_pred, prediction_label)

      $.ajax({
        url : "produce_label",
        data : {"id" : id, "image" : image, "xpos" : xpos, "ypos" : ypos},
        success : function(data) {
          all_data = JSON.parse(data);
          // Update node with assigned label
          total_samples = all_data['total_samples']
          document.getElementById("sample_total-out").innerHTML = total_samples;

          nodes[id]['label'] = all_data['label']
          nodes[id]['scatter_x'] = all_data['scatter_x']
          nodes[id]['scatter_y'] = all_data['scatter_y']
          nodes[id]['prediction_correct'] = true
          nodes[id]['prediction_confidence'] = max_pred
          document.getElementById("class-out").innerHTML = "Classification: " + all_data['label'];
          console.log(node)

          if (nodes[id]['label'] != label) {
            
            user_label_counter += 1
            console.log("Relabelling: ", label, " -> ", nodes[id]['label']);
            console.log("Label has changed! Counter: ", user_label_counter);
          }

          if (nodes[id]['label'] == prediction_label) {
            nodes[id]['prediction_correct'] = true
          } else {
            nodes[id]['prediction_correct'] = false
          }

          
          console.log('prediction_correct', nodes[id]['prediction_correct'])
          console.log('prediction_confidence', nodes[id]['prediction_confidence'])

          show_selected_node_in_scatter(node['scatter_id'], nodes[id]['label']);


        }
      });
      // node.x = null;
      // node.y = null;
    }
}
