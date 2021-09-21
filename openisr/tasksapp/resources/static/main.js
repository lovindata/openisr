// custom javascript

// On running do
(function() {
    // Disp a message on the console
    console.log('Sanity Check!');
})();

// These functions will be used on call
function handleClick(type) {
  // Request the route '/tasks' with the parameter {'type' : type}
  fetch('/tasks', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ type: type }),
  })
  // If the 'fetch' works then do
  .then(response => response.json())
  .then(data => {
    getStatus(data.task_id, true)
  })
}

function getStatus(taskID, firstCall) {
  // Request the route '/tasks/${taskID}'
  fetch(`/tasks/${taskID}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
    },
  })
  .then(response => response.json())
  .then(res => {
    // Disp on the console the obtained result
    console.log(res)

    // Build a HTML table line using the result
    const html = `
    <tr id=${taskID}>
      <td>${taskID}</td>
      <td>${res.task_status}</td>
      <td>${res.task_result}</td>
    </tr>`;
    if (firstCall === true) {
      // Get the HTML tag with the 'id=tasks'
      // 'insertRow' works with the tag specified as <tbody></tbody>
      // 'insertRow' prepares a line of <tbody></tbody> ready for modification
      const newRow = document.getElementById('tasks').insertRow(0);
      // Apply the modification
      newRow.innerHTML = html;
    } else {
      const existingRow = document.getElementById(`${taskID}`)
      // Apply the modification
      existingRow.innerHTML = html;
    }

    // If the status is not equal to 'PENDING' do 
    const taskStatus = res.task_status;
    // Return will stop the function execution
    // Everything after will not be runned
    if (taskStatus === 'SUCCESS' || taskStatus === 'FAILURE') return false; 

    // Run the recursively the function after 1 second
    // The setTimeout is an async function
    setTimeout(function() {
      getStatus(res.task_id, false);
    }, 1000);
  })

  // Catch the errors from the function 'then'
  .catch(err => console.log(err));
}