function create_bar_chart(bar_data, placement) {

  var margin = {
    top: 30,
    right: 40,
    bottom: 40,
    left: 50
  }

  var height = 200 - margin.top - margin.bottom,
    width = 385 - margin.left - margin.right,
    barWidth = 50,
    barOffset = 5;

  var svg = d3.select(placement).append('svg');

  var data = [bar_data[0]['batchsize'], bar_data[0]['testsize'], bar_data[0]['poolsize']]
  console.log(data)

  var tempColour;

  var colours = d3.scaleLinear()
    .domain([0, data.length]) // Dependant on value, colour will change.
    .range(['#FFB832', '#C61C6F'])

  var xScale = d3.scaleBand()
    // .domain(d3.range(0, data.length)) // Remap values into rangebands
    .domain(d3.range(0, 3))
    .range([0, width], 0.1, 0.5) // from 0 to width of the background.


  var yScale = d3.scaleLinear() // Remap to svg background
    .domain([0, d3.max(data)]) //set domain to max value in seq.
    .range([0, height]) // height of chart max.

  var vGuideScale = d3.scaleLinear()
    .domain([0, d3.max(data)])
    .range([height, 0])

  var vAxis = d3.axisLeft(vGuideScale)
    .ticks(10)

  var vGuide = svg.append('g')
  vAxis(vGuide)
  vGuide.attr('transform', 'translate(' + margin.left + ', ' + margin.top + ')')
  vGuide.selectAll('path')
    .style({
      fill: 'none',
      stroke: '#000'
    })
  vGuide.selectAll('line')
    .style({
      stroke: '#000'
    })

  var hAxis = d3.axisBottom(xScale)
    // .ticks(data.size)
    .ticks(3)

  var hGuide = svg.append('g')
  hAxis(hGuide)
  hGuide.attr('transform', 'translate(' + margin.left + ', ' + (height + margin.top) + ')')
  hGuide.selectAll('path')
    .style({
      fill: 'none',
      stroke: '#000'
    })
  hGuide.selectAll('line')
    .style({
      stroke: '#000'
    })

  var tooltip = d3.select(placement).append('div')
    .style('postion', 'absolute')
    .style('padding', '0 10px')
    .style('background', 'white')
    .style('opacity', 0)

  var barChart = svg
    .style('background', '#E7E0CB')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .append('g')
    .attr('transform', 'translate(' + margin.left + ', ' + margin.top + ')')
    .selectAll('rect').data(data)
    .enter().append('rect')
    .style('fill', function(d, i) {
      return colours(i);
    })
    .attr('width', xScale.bandwidth())
    .attr('x', function(d, i) {
      return xScale(i);
    })
    .attr('height', 0)
    .attr('y', height)

    .on('mouseover', function(d) { // Mouse events
      //tool tips don't work?
      tooltip.transition()
        .style('opacity', .9)

      tooltip.html(d)
        .style('left', (d3.event.pageX) + 'px')
        .style('top', (d3.event.pageY) + 'px')

      tempColour = this.style.fill;
      d3.select(this)
        .style('opacity', .5)
        .style('fill', 'red')
    })
    .on('mouseout', function(d) {
      d3.select(this)
        .style('opacity', 1)
        .style('fill', tempColour)
    })

  barChart.transition()
    .attr('height', function(d) {
      return yScale(d);
    })
    .attr('y', function(d) {
      return height - yScale(d); // from bottom of svg.
    })
    .delay(function(d, i) {
      return i * 10; // time to bars
    })
    .duration(800)
    .ease(d3.easeElastic)
}

function update_bar_chart(new_data) {

  var margin = {
    top: 30,
    right: 40,
    bottom: 40,
    left: 50
  }

  var height = 200 - margin.top - margin.bottom,
    width = 385 - margin.left - margin.right,
    barWidth = 50,
    barOffset = 5;

  var barchart = d3.select("#samples_barchart_view").transition();

  var yScale = d3.scaleLinear() // Remap to svg background
    .domain([0, d3.max(new_data)]) //set domain to max value in seq.
    .range([0, height]) // height of chart max.

  var bars = barchart.selectAll("rect").data(new_data)
    .enter().append('rect')
    .style('fill', function(d, i) {
      return colours(i);
    })
    .attr("height", function(d) {
      return yScale(d);
    })
    .attr('y', function(d) {
      return height - yScale(d);
    });

  bars.exit().remove();

}


// bars.attr("class", "update");
//
// bars.enter().append("rect")
//   .attr("class", "enter")
//   .attr("height", function(d) {
//     return yScale(d);
//   })
//   .attr('y', function(d) {
//     return height - yScale(d);
//   });
// bars.exit().remove();
// }
