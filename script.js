
function getPhysicalShape(shape, tiling) {
    assert(shape.length == tiling.length)

    let physical_shape = Array(shape.length);
    for (let i = 0; i < shape.length; i++) {
        physical_shape[i] = roundUpToMultiple(shape[i], tiling[i]);
    }
    return physical_shape;
}

function create1DTable(tableData, tableId, containerId) {
    let table = document.createElement("table");
    table.id = tableId;
    let tableBody = document.createElement("tbody");

    console.log(tableData);

    let row = document.createElement('tr');
    tableData.forEach(function (rowData) {

        let cell = document.createElement("td");
        if (rowData == -1) {
            cell.classList.add("pad-value")
        }
        cell.appendChild(document.createTextNode(rowData));
        row.appendChild(cell);

    });
    tableBody.appendChild(row);

    table.appendChild(tableBody);

    // If there's an existing table with this ID, replace it.
    // Otherwise just add it.
    let oldTable = document.getElementById(tableId);
    if (oldTable == undefined) {
        let table_container = document.getElementById(containerId);
        table_container.appendChild(table);
    } else {
        oldTable.replaceWith(table);
    }
}

function create2DTable(tableData, tableId, containerId) {
    let table = document.createElement("table");
    table.id = tableId;
    let tableBody = document.createElement("tbody");

    console.log(tableData);

    tableData.forEach(function (rowData) {
        let row = document.createElement('tr');
        rowData.forEach(function (cellData) {
            let cell = document.createElement("td");
            if (cellData == -1) {
                cell.classList.add("pad-value")
            }
            cell.appendChild(document.createTextNode(cellData));
            row.appendChild(cell);
        });
        tableBody.appendChild(row);
    });

    table.appendChild(tableBody);

    // If there's an existing table with this ID, replace it.
    // Otherwise just add it.
    let oldTable = document.getElementById(tableId);
    if (oldTable == undefined) {
        let table_container = document.getElementById(containerId);
        table_container.appendChild(table);
    } else {
        oldTable.replaceWith(table);
    }
}

function assert(condition, message) {
    if (!condition) {
        throw new Error(message || "Assertion failed");
    }
}

function Product(array) {
    return array.reduce((accumulator, currentValue) => accumulator * currentValue, 1);
}

function roundUpToMultiple(number, multiple) {
    return Math.ceil(number / multiple) * multiple;
}

// See: https://www.tensorflow.org/xla/tiled_layout
// linear_index(e, d)
//     = linear_index((en, en-1, ... , e1), (dn, dn-1, ... , d1))
//     = endn-1...d1 + en-1dn-2...d1 + ... + e1
function get_linear_index(e, d) {
    assert(e.length == d.length, "index and shape must have same rank");
    let rank = e.length;
    linear_index = 0;
    for (let i = 0; i < rank; i++) {
        prod = e[i];
        for (let j = (i + 1); j < rank; j++) {
            prod = prod * d[j];
        }
        linear_index = linear_index + prod;
    }

    return linear_index;
}

// See: https://www.tensorflow.org/xla/tiled_layout
// linear_index_with_tile(e, d, t)
//     = linear_index((⌊e/t⌋, e mod t), (⌈d/t⌉, t))     (arithmetic is elementwise, (a,b) is concatenation)
//     = linear_index((⌊en/tn⌋, ... , ⌊e1/t1⌋, en mod tn, ... , e1 mod t1), (⌈dn/tn⌉, ... , ⌈d1/t1⌉, tn, tn-1, ... , t1))
//     = linear_index(
//           (⌊en/tn⌋, ... , ⌊e1/t1⌋), (⌈dn/tn⌉, ... , ⌈d1/t1⌉)) ∙ tntn-1...t1 
//               + linear_index((en mod tn, ... , e1 mod t1), (tn, tn-1, ... , t1))
function get_linear_index_with_tiling(e, d, t) {
    assert(e.length == d.length, "index and shape must have same rank");
    // TODO(joshvarty): We can relax this.
    assert(e.length == t.length, "index and tiling must have same rank");

    let rank = e.length;

    let prefix = [];
    for (let i = 0; i < rank; i++) {
        prefix.push(Math.floor(e[i] / t[i]));
    }

    for (let i = 0; i < rank; i++) {
        prefix.push(e[i] % t[i]);
    }

    let suffix = [];
    for (let i = 0; i < rank; i++) {
        suffix.push(Math.ceil(d[i] / t[i]));
    }

    for (let i = 0; i < rank; i++) {
        suffix.push(t[i]);
    }

    return get_linear_index(prefix, suffix);
}

// Generate the 1-D memory that backs our shape including any padding.
function generateData(shape, tiling) {
    // TODO(joshvarty): Support tilings that do not have identical
    // rank as the tensors they are applied to.
    assert(shape.length == tiling.length, "shape and tiling must have same rank");

    let roundedRows = roundUpToMultiple(shape[0], tiling[0]);
    let roundedCols = roundUpToMultiple(shape[1], tiling[1]);

    let physical_data = Array(Product([roundedRows, roundedCols])).fill(-1);
    let logical_data = Array(roundedRows).fill().map(() => Array(roundedCols).fill(-1));

    let counter = 0;
    for (let i = 0; i < shape[0]; i++) {
        for (let j = 0; j < shape[1]; j++) {
            let multidimensional_index = [i, j];
            let linear_index = get_linear_index_with_tiling(multidimensional_index, shape, tiling);
            physical_data[linear_index] = counter;
            logical_data[i][j] = counter;
            counter = counter + 1;
        }
    }

    // The 2D representation of the 1D physical data if we ignore tiling altogether.
    let memref_data = Array(roundedRows).fill().map(() => Array(roundedCols).fill(-1));
    let physical_shape = getPhysicalShape(shape, tiling);
    for (let i = 0; i < physical_shape[0]; i++) {
        for (let j = 0; j < physical_shape[1]; j++) {
            let linear_index = i * physical_shape[1] + j;
            memref_data[i][j] = physical_data[linear_index];
        }
    }

    return [physical_data, logical_data, memref_data];
}

function parseXlaShape(shapeString) {
    const pattern = /^([a-z]+\d+)\[([\d,]+)\]\{([\d,]+):t\(([\d,]+)\)\}$/;
    const match = shapeString.match(pattern);

    if (!match) {
        throw new Error("Invalid XLA shape format");
    }

    const dataType = match[1];
    const logicalDimensions = match[2].split(',').map(Number);
    const layoutDimensions = match[3].split(',').map(Number);
    const tilingDimensions = match[4].split(',').map(Number);

    return {
        dataType,
        logicalDimensions,
        layoutDimensions,
        tilingDimensions,
    };
}

function visualizeShape() {
    let rawShapeText = document.getElementById('shape-input').value.toLowerCase();
    console.log(rawShapeText);

    const { dataType, logicalDimensions, layoutDimensions, tilingDimensions } = parseXlaShape(rawShapeText);
    document.getElementById("datatype").innerText = dataType;
    document.getElementById("logical-dimensions").innerText = logicalDimensions;
    document.getElementById("layout-dimensions").innerText = layoutDimensions;
    document.getElementById("tiling-dimensions").innerText = tilingDimensions;

    if (logicalDimensions.length != 2) {
        alert("Unsupported number of logical dimensions: " + logicalDimensions.length);
        return;
    }

    if (tilingDimensions.length != 2) {
        alert("Unsupported number of tiling dimensions: " + tilingDimensions.length);
        return;
    }

    console.log(logicalDimensions);
    console.log(tilingDimensions);

    // TODO(joshvarty): Parse from input shape.
    let rows = logicalDimensions[0];
    let cols = logicalDimensions[1];
    let shape = [rows, cols];

    let tile0 = tilingDimensions[0];
    let tile1 = tilingDimensions[1];
    let tiling = [tile0, tile1];

    const [physical_data, logical_data, memref_data] = generateData(shape, tiling);
    console.log(physical_data);
    console.log(logical_data);

    let logical_view = create2DTable(logical_data, "table2d", "logical-view");
    let physical_view = create1DTable(physical_data, "table1d", "physical-view");
    let memmref_view = create2DTable(memref_data, "table2d-rename-me", "logical-view-of-physical-view");

}

// Set up click event for visualization button.
document.getElementById("visualize-button").onclick = function () {
    visualizeShape();
};

// Set up KeyUp handler to visualize when the user presses <Enter>.
document.getElementById("shape-input").onkeyup = function (e) {
    if (e.key === 'Enter' || e.keyCode === 13) {
        visualizeShape();
    }
}

document.addEventListener("DOMContentLoaded", function () {
    // Start with the default shape.
    visualizeShape();
});
