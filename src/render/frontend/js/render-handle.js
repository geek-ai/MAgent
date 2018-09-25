var _gridCTX = undefined;
var _staticCTX = undefined;
var _agentRangeCTX = undefined;
var _agentBodyCTX = undefined;
var _agentHPCTX = undefined;
var _eventCTX = undefined;
var _statusStaticCTX = undefined;
var _minimapCTX = undefined;
var _minimapPostionCTX = undefined;

var _socket = undefined;
var _offsetX = undefined;
var _offsetY = undefined;
var _mapFPS = undefined;
var _mapIPS = undefined;
var _mapProcessedFrame = undefined;
var _mapProcessedImage = undefined;
var _mapCurrentImage = undefined;
var _mapTotalImage = undefined;
var _mapStatTime = undefined;
var _mapAnimateTick = undefined;
var _mapStatus = undefined;
var _mapData = undefined;
var _mapLastData = undefined;
var _mapStyles = undefined;
var _mapSpeed = undefined;
var _mapForcedPause = undefined;

var _isBrowserSizeChanged = undefined;
var _isWindowChanged = undefined;
var _isGridSizeChanged = undefined;

function _onresize() {
    _isBrowserSizeChanged = true;
    _isGridSizeChanged = true;
    _isWindowChanged = true;
}

function _drawGrid() {
    /*_gridCTX.clearRect(0, 0, _gridCTX.canvas.width, _gridCTX.canvas.height);
    %_gridCTX.beginPath();
    _gridCTX.strokeStyle = gridColor;
    var i;
    var magentX = Math.floor(_offsetX) - _offsetX;
    var magentY = Math.floor(_offsetY) - _offsetY;
    for (i = 0; i * gridSize < _gridCTX.canvas.height; i++) {
        _gridCTX.moveTo(0, i * gridSize + magentY / gridSize);
        _gridCTX.lineTo(_gridCTX.canvas.width, i * gridSize + magentY / gridSize);
    }
    for (i = 0; i * gridSize < _gridCTX.canvas.width; i++) {
        _gridCTX.moveTo(i * gridSize + magentX / gridSize, 0);
        _gridCTX.lineTo(i * gridSize + magentX / gridSize, _gridCTX.canvas.height);
    }
    _gridCTX.stroke();*/
}

function _drawStatusFigure() {
    _statusStaticCTX.clearRect(0, 0, _statusStaticCTX.canvas.width, _statusStaticCTX.canvas.height);
    _statusStaticCTX.beginPath();
    if (_socket !== undefined) {
        if (_socket.readyState === _socket.CONNECTING) {
            _statusStaticCTX.strokeText('render Status: CONNECTING', STATUS_PADDING_LEFT, STATUS_PADDING_TOP);
        } else if (_socket.readyState === _socket.OPEN) {
            _statusStaticCTX.strokeText('render Status: OPEN', STATUS_PADDING_LEFT, STATUS_PADDING_TOP);
            if (_mapStyles !== undefined) {
                _statusStaticCTX.strokeText('IPS: ' + _mapIPS.toString(), STATUS_PADDING_LEFT, STATUS_PADDING_TOP + STATUS_SPACING * 2);
                _statusStaticCTX.strokeText('Window: ('
                    + Math.floor(_offsetY).toString() + ', '
                    + Math.floor(_offsetY).toString() + ', '
                    + (Math.ceil(_offsetX + _gridCTX.canvas.width / gridSize)).toString() + ', '
                    + (Math.ceil(_offsetY + _gridCTX.canvas.height / gridSize)).toString() + ')',
                    STATUS_PADDING_LEFT, STATUS_PADDING_TOP + STATUS_SPACING * 3
                );
            }
        } else if (_socket.readyState === _socket.CLOSING) {
            _statusStaticCTX.strokeText('render Status: CLOSING', STATUS_PADDING_LEFT, STATUS_PADDING_TOP);
        } else {
            _statusStaticCTX.strokeText('render Status: CLOSED', STATUS_PADDING_LEFT, STATUS_PADDING_TOP);
        }
    } else {
        _statusStaticCTX.strokeText('render Status: PREPARING', STATUS_PADDING_LEFT, STATUS_PADDING_TOP);
    }
    _statusStaticCTX.strokeText('FPS: ' + _mapFPS.toString(), STATUS_PADDING_LEFT, STATUS_PADDING_TOP + STATUS_SPACING);
}

function _drawNumbers() {
    _statusDynamicCTX.clearRect(0, 0, _statusDynamicCTX.canvas.width, _statusDynamicCTX.canvas.height);
    for (var i = 0; i < _mapData[5].length; i++) {
        _statusDynamicCTX.strokeText(
            'Group ' + i.toString() + ' number: ' + _mapData[5][i],
            STATUS_PADDING_LEFT,
            STATUS_PADDING_TOP + STATUS_SPACING * (4 + i)
        );
    }
}

function _onkeydown(event) {
    var windowCenterX, windowCenterY;
    if (event.keyCode === 37) {
        // Left
        _offsetX = _offsetX - Math.max(1, Math.round(MOVE_SPACING * 10 / gridSize));
        _isWindowChanged = true;
    } else if (event.keyCode === 38) {
        // Up
        _offsetY = _offsetY - Math.max(1, Math.round(MOVE_SPACING * 10 / gridSize));
        _isWindowChanged = true;
    } else if (event.keyCode === 39) {
        // Right
        _offsetX = _offsetX + Math.max(1, Math.round(MOVE_SPACING * 10 / gridSize));
        _isWindowChanged = true;
    } else if (event.keyCode === 40) {
        // Down
        _offsetY = _offsetY + Math.max(1, Math.round(MOVE_SPACING * 10 / gridSize));
        _isWindowChanged = true;
    } else if (event.keyCode === 69) {
        // E: Edit file name
        $("#magnet-file-modal").modal('show');
    } else if (event.keyCode === 72) {
        // H: Help
        $("#magnet-help-modal").modal('show');
    } else if (event.keyCode === 83) {
        // S: Settings
        $("#magnet-settings-modal").modal('show');
    } else if (event.keyCode === 188) {
        // ,<: Zoom Out
        windowCenterX = _gridCTX.canvas.width / 2 / gridSize + _offsetX;
        windowCenterY = _gridCTX.canvas.height / 2 / gridSize + _offsetY;
        gridSize = Math.max(1, gridSize - 1);
        _offsetX = _offsetX + windowCenterX - (_gridCTX.canvas.width / 2 / gridSize + _offsetX);
        _offsetY = _offsetY + windowCenterY - (_gridCTX.canvas.height / 2 / gridSize + _offsetY);
        _isGridSizeChanged = true;
        _isWindowChanged = true;
    } else if (event.keyCode === 190) {
        // .>: Zoom In
        windowCenterX = _gridCTX.canvas.width / 2 / gridSize + _offsetX;
        windowCenterY = _gridCTX.canvas.height / 2 / gridSize + _offsetY;
        gridSize = Math.min(100, gridSize + 1);
        _offsetX = _offsetX + windowCenterX - (_gridCTX.canvas.width / 2 / gridSize + _offsetX);
        _offsetY = _offsetY + windowCenterY - (_gridCTX.canvas.height / 2 / gridSize + _offsetY);
        _isGridSizeChanged = true;
        _isWindowChanged = true;
    } else if (event.keyCode === 80) {
        // P: Pause
        _mapForcedPause ^= 1;
    }
}

function _collectTicks() {
    var timenow = performance.now();
    var elapsed = timenow - _mapStatTime;
    _mapIPS = elapsed < STATUS_FPS_PERIOD ? 'N/A' : Math.round(1000 * _mapProcessedImage / elapsed);
    _mapFPS = elapsed < STATUS_FPS_PERIOD ? 'N/A' : Math.round(1000 * _mapProcessedFrame / elapsed);
    _mapStatTime = timenow;
    _mapProcessedFrame = 0;
    _mapProcessedImage = 0;
    _drawStatusFigure();
    setTimeout(_collectTicks, STATUS_FPS_PERIOD);
}

function run() {
    _statusStaticCTX = document.getElementById('magnet-canvas-status-physical-information').getContext('2d');
    _statusStaticCTX.canvas.height = STATUS_HEIGHT;
    _statusStaticCTX.canvas.width = STATUS_WIDTH;
    _statusDynamicCTX = document.getElementById('magnet-canvas-status-statistics').getContext('2d');
    _statusDynamicCTX.canvas.height = STATUS_HEIGHT;
    _statusDynamicCTX.canvas.width = STATUS_WIDTH;

    _mapStatTime = performance.now();

    _collectTicks();

    window.addEventListener('resize', _onresize);
    window.addEventListener('keydown', _onkeydown);
    $('#magnet-file-form-submit').click(function (e) {
        e.preventDefault();
        if (_socket.readyState !== _socket.OPEN) {
            $.jGrowl('This client is not connected to the server, please waiting to be connected', {
                position: 'bottom-right'
            });
        } else {
            _mapStatus = 'STOP';
            _socket.send('l' + $('#magnet-file-form-conf').val() + ',' + $('#magnet-file-form-map').val());
            $('#magnet-file-modal').modal('hide');
        }
    });
    $('#magnet-file-modal').on('shown.bs.modal', function() {
        window.removeEventListener('keydown', _onkeydown);
    }).on('hide.bs.modal', function() {
        window.addEventListener('keydown', _onkeydown);
    });
    $('#magnet-help-modal').on('shown.bs.modal', function() {
        window.removeEventListener('keydown', _onkeydown);
    }).on('hide.bs.modal', function() {
        window.addEventListener('keydown', _onkeydown);
    });
    $('#magnet-settings-modal').on('shown.bs.modal', function() {
        window.removeEventListener('keydown', _onkeydown);
    }).on('hide.bs.modal', function() {
        window.addEventListener('keydown', _onkeydown);
    });


    var _connect = function () {
        _socket = new WebSocket(SOCKET_HOST);
        _socket.onopen = function () {
            console.log('successfully connected to the backend server');
            $("#magnet-file-modal").modal('show');
            _drawStatusFigure();
        };
        _socket.onclose = function () {
            console.log('connection closed.');
            _mapStatus = 'STOP';
            $.jGrowl('Connection lost from the server.', {
                position: 'bottom-right'
            });
            $('#magnet-settings-progress').unbind().attr('disabled', true);
            $('#magnet-settings-speed').unbind().attr('disabled', true);
            setTimeout(_connect, SOCKET_RECONNECT_PERIOD);
            _drawStatusFigure();
        };
        _socket.onerror = function () {
            console.log('connection lost.');
            _drawStatusFigure();
        };
        _socket.onmessage = function (data) {
            data = data.data;
            var op = data[0];
            data = data.substr(1);
            switch (op) {
                case 'i':
                    var pos = data.indexOf('|');
                    _mapStyles = eval('(' + data.substr(pos + 1) + ')');
                    _mapTotalImage = parseInt(data.substr(0, pos));
                    _mapProcessedImage = 0;
                    _mapCurrentImage = 0;
                    _mapLastData = undefined;
                    _mapData = undefined;
                    _offsetX = 0;
                    _offsetY = 0;
                    gridSize = 10;
                    _mapAnimateTick = 0;
                    _mapSpeed = 250;
                    _mapForcedPause = false;

                    _gridCTX = document.getElementById('magnet-canvas-grid').getContext('2d');
                    _agentRangeCTX = document.getElementById('magnet-canvas-agent-range').getContext('2d');
                    _eventCTX = document.getElementById('magnet-canvas-event').getContext('2d');
                    _agentBodyCTX = document.getElementById('magnet-canvas-agent-body').getContext('2d');
                    _staticCTX = document.getElementById('magnet-canvas-static').getContext('2d');
                    _agentHPCTX = document.getElementById('magnet-canvas-agent-hp').getContext('2d');
                    _minimapCTX = document.getElementById('magnet-canvas-minimap').getContext('2d');
                    _minimapCTX.canvas.height = _mapStyles['minimap-height'];
                    _minimapCTX.canvas.width = _mapStyles['minimap-width'];
                    _minimapPostionCTX = document.getElementById('magnet-canvas-minimap-position').getContext('2d');
                    _minimapPostionCTX.canvas.height = _mapStyles['minimap-height'];
                    _minimapPostionCTX.canvas.width = _mapStyles['minimap-width'];

                    _onresize();
                    $('#magnet-settings-progress')
                        .removeAttr('disabled')
                        .attr('min', 0)
                        .attr('max', _mapTotalImage - 1)
                        .attr('value', 0)
                        .change(function () {
                            _mapLastData = undefined;
                            _mapData = undefined;
                            _mapAnimateTick = 0;
                            _mapCurrentImage = parseInt($('#magnet-settings-progress').val());
                            _socket.send('p' + _mapCurrentImage.toString() + ' '
                                + Math.floor(_offsetX).toString() + ' ' + Math.floor(_offsetY).toString() + ' '
                                + Math.ceil(_offsetX + window.innerWidth / gridSize).toString() + ' '
                                + Math.ceil(_offsetY + window.innerHeight / gridSize).toString());
                            _mapStatus = 'PAUSE';
                        })
                        .bind('slide', function () {
                            _mapLastData = undefined;
                            _mapData = undefined;
                            _mapAnimateTick = 0;
                            _mapCurrentImage = parseInt($('#magnet-settings-progress').val());

                            _socket.send('p' + _mapCurrentImage.toString() + ' '
                                + Math.floor(_offsetX).toString() + ' ' + Math.floor(_offsetY).toString() + ' '
                                + Math.ceil(_offsetX + window.innerWidth / gridSize).toString() + ' '
                                + Math.ceil(_offsetY + window.innerHeight / gridSize).toString());
                            _mapStatus = 'PAUSE';
                        });
                    $('#magnet-settings-speed')
                        .removeAttr('disabled')
                        .attr('min', 0)
                        .attr('max', ANIMATE_STEP)
                        .attr('value', 250)
                        .bind('change', function () {
                            _mapSpeed = parseInt($('#magnet-settings-speed').val());
                        });
                    _mapStatus = 'PAUSE';

                    _animate();

                    _socket.send('p' + _mapCurrentImage.toString() + ' '
                        + Math.floor(_offsetX).toString() + ' ' + Math.floor(_offsetY).toString() + ' '
                        + Math.ceil(_offsetX + window.innerWidth / gridSize).toString() + ' '
                        + Math.ceil(_offsetY + window.innerHeight / gridSize).toString());
                    break;
                case 'f':
                    _mapStatus = 'PLAY';
                    _mapLastData = _mapData;
                    _mapData = [[], [], [], [], [], []];

                    data = data.split(';');

                    data[0] = data[0].split('|'); // Events
                    for (var itEvents = 0, nEvents = data[0].length; itEvents < nEvents; itEvents++) {
                        if (data[0][itEvents] !== '') {
                            data[0][itEvents] = data[0][itEvents].split(' ');
                            _mapData[0].push([
                                parseInt(data[0][itEvents][0]),
                                parseInt(data[0][itEvents][1]),
                                parseInt(data[0][itEvents][2]),
                                parseInt(data[0][itEvents][3])
                            ]);
                        }
                    }

                    data[1] = data[1].split('|'); // Agents
                    for (var itAgents = 0, nAgents = data[1].length; itAgents < nAgents; itAgents++) {
                        if (data[1][itAgents] === '') continue;
                        data[1][itAgents] = data[1][itAgents].split(' ');
                        var x = parseInt(data[1][itAgents][1]);
                        var y = parseInt(data[1][itAgents][2]);
                        var group = parseInt(data[1][itAgents][3]);
                        var dir = parseInt(data[1][itAgents][4]);
                        var hp = parseInt(data[1][itAgents][5]);
                        if (!_mapStyles['group'].hasOwnProperty(group)) {
                            $.jGrowl('group ' + group.toString() + ' is not found in the configuration file', {
                                position: 'bottom-right'
                            });
                            _mapStatus = 'STOP';
                            break;
                        }
                        if (dir === 90) {
                            x = x + _mapStyles['group'][group]['height'] - 1;
                            y = y + _mapStyles['group'][group]['width'] - 1;
                        } else if (dir === 180) {
                            y = y + _mapStyles['group'][group]['height'] - 1;
                        } else if (dir === 0) {
                            x = x + _mapStyles['group'][group]['width'] - 1;
                        }
                        _mapData[1][data[1][itAgents][0]] = [x, y, group, dir, hp]
                    }

                    data[2] = data[2].split('|');
                    for (var itBreads = 0, nBreads = data[2].length; itBreads < nBreads; itBreads++) {
                        if (data[2][itBreads] !== '') {
                            data[2][itBreads] = data[2][itBreads].split(' ');
                            _mapData[2].push([
                                parseInt(data[2][itBreads][0]),
                                parseInt(data[2][itBreads][1]),
                                parseInt(data[2][itBreads][2])
                            ]);
                        }
                    }
                    data[3] = data[3].split('|');
                    for (var itObstacles = 0, nObstacles = data[3].length; itObstacles < nObstacles; itObstacles++) {
                        if (data[3][itObstacles] !== '') {
                            data[3][itObstacles] = data[3][itObstacles].split(' ');
                            _mapData[3].push([
                                parseInt(data[3][itObstacles][0]),
                                parseInt(data[3][itObstacles][1])
                            ]);
                        }
                    }
                    data[4] = data[4].split(' ');
                    for (var itPixels = 0, nPixels = data[4].length; itPixels < nPixels; itPixels++) {
                        if (data[4][itPixels] !== '') {
                            _mapData[4].push(parseInt(data[4][itPixels]));
                        }
                    }

                    _mapData[5] = data[5].split(' ');
                    break;
                case 'e':
                    $.jGrowl(data, {position: 'bottom-right'});
                    break;
                default:
                    $.jGrowl('invalid message from backend. Please check the version of frontend and backend', {
                        position: 'bottom-right'
                    });
                    break;
            }
        };
    };
    _connect();
}

function _getOriginGridCoordinate(curData, oldData, style) {
    var theta, anchorX, anchorY;
    var rate = Math.min(_mapAnimateTick, ANIMATE_STEP) / ANIMATE_STEP;
    var nowAngle = (curData[3] + 90) / 180 * Math.PI;
    var nowAnchorX = curData[0] + style['anchor'][0] * Math.cos(nowAngle) - style['anchor'][1] * Math.sin(nowAngle) - _offsetX;
    var nowAnchorY = curData[1] + style['anchor'][0] * Math.sin(nowAngle) + style['anchor'][1] * Math.cos(nowAngle) - _offsetY;

    if (oldData !== undefined) {
        var lastAngle = (oldData[3] + 90) / 180 * Math.PI;
        if (Math.abs(lastAngle - nowAngle) < Math.PI) {
            theta = (nowAngle - lastAngle) * rate + lastAngle;
        } else if (lastAngle > nowAngle) {
            theta = (nowAngle - lastAngle + Math.PI * 2) * rate + lastAngle;
        } else {
            theta = (nowAngle - lastAngle - Math.PI * 2) * rate + lastAngle;
        }
        var preAnchorX = oldData[0] + style['anchor'][0] * Math.cos(lastAngle) - style['anchor'][1] * Math.sin(lastAngle) - _offsetX;
        var preAnchorY = oldData[1] + style['anchor'][0] * Math.sin(lastAngle) + style['anchor'][1] * Math.cos(lastAngle) - _offsetY;

        anchorX = nowAnchorX * rate + preAnchorX * (1 - rate);
        anchorY = nowAnchorY * rate + preAnchorY * (1 - rate);
    } else {
        theta = nowAngle;
        anchorX = nowAnchorX;
        anchorY = nowAnchorY;
    }
    var originXMaster =  anchorX * Math.cos(theta) + anchorY * Math.sin(theta) - style['anchor'][0] + Math.sqrt(2) / 2 * Math.cos(Math.PI / 4 - theta) - 0.5;
    var originYMaster = -anchorX * Math.sin(theta) + anchorY * Math.cos(theta) - style['anchor'][1] + Math.sqrt(2) / 2 * Math.sin(Math.PI / 4 - theta) - 0.5;
    return [originXMaster, originYMaster, theta];
}

function _drawAgent() {
    var counter = 0;
    _agentBodyCTX.clearRect(0, 0, _agentBodyCTX.canvas.width, _agentBodyCTX.canvas.height);
    _agentRangeCTX.clearRect(0, 0, _agentRangeCTX.canvas.width, _agentRangeCTX.canvas.height);
    _agentHPCTX.clearRect(0, 0, _agentHPCTX.canvas.width, _agentHPCTX.canvas.height);
    for (var agentID in _mapData[1]) {
        if (_mapData[1].hasOwnProperty(agentID)) {
            var style = _mapStyles['group'][_mapData[1][agentID][2]];
            var result = _getOriginGridCoordinate(
                _mapData[1][agentID],
                _mapLastData !== undefined ? _mapLastData[1][agentID] : undefined,
                style
            );
            var originXMaster = result[0];
            var originYMaster = result[1];

            _agentBodyCTX.beginPath();
            _agentBodyCTX.rotate(result[2]);
            _agentBodyCTX.fillStyle = style['style'];
            _agentBodyCTX.rect(
                originXMaster * gridSize,
                originYMaster * gridSize,
                style['height'] * gridSize,
                style['width'] * gridSize
            );
            _agentBodyCTX.fill();
            _agentBodyCTX.rotate(-result[2]);

            if (gridSize >= 4) {
                //if (parseInt(style['vision-angle']) <= 180) {
                //    _agentRangeCTX.beginPath();
                //    _agentRangeCTX.fillStyle = style['vision-style'];
                //    _agentRangeCTX.rotate(result[2]);
                //    _agentRangeCTX.moveTo(
                //        originXMaster * gridSize + style['height'] * gridSize / 2,
                //        originYMaster * gridSize + style['width'] * gridSize / 2
                //    );
                //    _agentRangeCTX.arc(
                //        originXMaster * gridSize + style['height'] * gridSize / 2,
                //        originYMaster * gridSize + style['width'] * gridSize / 2,
                //        style['vision-radius'] * gridSize,
                //        -(parseInt(style['vision-angle'])) / 360 * Math.PI - Math.PI / 2,
                //        (parseInt(style['vision-angle'])) / 360 * Math.PI - Math.PI / 2,
                //        false
                //    );
                //    _agentRangeCTX.fill();
                //    _agentRangeCTX.rotate(-result[2]);
                //}

                _agentRangeCTX.beginPath();
                _agentRangeCTX.fillStyle = style['attack-style'];
                _agentRangeCTX.rotate(result[2]);
                _agentRangeCTX.moveTo(
                    originXMaster * gridSize + style['height'] * gridSize / 2,
                    originYMaster * gridSize + style['width'] * gridSize / 2
                );
                _agentRangeCTX.arc(
                    originXMaster * gridSize + style['height'] * gridSize / 2,
                    originYMaster * gridSize + style['width'] * gridSize / 2,
                    style['attack-radius'] * gridSize,
                    -(parseInt(style['attack-angle'])) / 360 * Math.PI - Math.PI / 2,
                    (parseInt(style['attack-angle'])) / 360 * Math.PI - Math.PI / 2,
                    false
                );
                _agentRangeCTX.fill();
                _agentRangeCTX.rotate(-result[2]);

                if (gridSize >= 6) {
                    _agentHPCTX.beginPath();
                    _agentHPCTX.rotate(result[2]);
                    _agentHPCTX.rect(
                        originXMaster * gridSize,
                        originYMaster * gridSize,
                        gridSize / 4,
                        style['width'] * gridSize
                    );
                    _agentHPCTX.strokeStyle = "rgba(0,0,0,1)";
                    //_agentHPCTX.stroke();
                    _agentHPCTX.fillStyle = "rgba(255,255,255,1)";
                    _agentHPCTX.fill();

                    _agentHPCTX.beginPath();
                    _agentHPCTX.fillStyle = style['style'];
                    var hp;
                    var rate = Math.min(_mapAnimateTick, ANIMATE_STEP) / ANIMATE_STEP;
                    if (_mapLastData !== undefined && _mapLastData[1].hasOwnProperty(agentID)) {
                        hp = _mapLastData[1][agentID][4] * (1 - rate) + _mapData[1][agentID][4] * rate;
                    } else {
                        hp = _mapData[1][agentID][4];
                    }
                    _agentHPCTX.rect(
                        originXMaster * gridSize,
                        originYMaster * gridSize + (100 - hp) / 100 * style['width'] * gridSize,
                        gridSize / 4,
                        hp / 100 * style['width'] * gridSize
                    );
                    _agentHPCTX.fill();
                    _agentHPCTX.rotate(-result[2]);
                }
            }
        }
    }
}

function _drawObstacles() {
    _staticCTX.clearRect(0, 0, _staticCTX.canvas.width, _staticCTX.canvas.height);
    _staticCTX.beginPath();
    _staticCTX.fillStyle = _mapStyles['obstacle-style'];
    for (var i = 0; i < _mapData[3].length; i++) {
        _staticCTX.rect(
            (_mapData[3][i][0] - _offsetX) * gridSize,
            (_mapData[3][i][1] - _offsetY) * gridSize,
            gridSize, gridSize
        );
    }
    _staticCTX.fill();
}

function _drawEvent() {
    _eventCTX.clearRect(0, 0, _eventCTX.canvas.width, _eventCTX.canvas.height);
    if (gridSize >= 4) {
        _eventCTX.beginPath();
        _eventCTX.strokeStyle = _mapStyles['attack-style'];
        for (var i = 0; i < _mapData[0].length; i++) {
            if (_mapData[0][i][0] === 0) {
                var id = _mapData[0][i][1];
                if (_mapData[1][id] === undefined) {
                    continue;
                }
                var style = _mapStyles['group'][_mapData[1][id][2]];
                var result = _getOriginGridCoordinate(
                    _mapData[1][id],
                    _mapLastData !== undefined ? _mapLastData[1][id] : undefined,
                    style
                );
                _eventCTX.rotate(result[2]);
                _eventCTX.moveTo(
                    result[0] * gridSize + style['height'] * gridSize / 2,
                    result[1] * gridSize + style['width'] * gridSize / 2
                );
                _eventCTX.rotate(-result[2]);
                _eventCTX.lineTo(
                    _mapData[0][i][2] * gridSize + gridSize / 2 - _offsetX * gridSize,
                    _mapData[0][i][3] * gridSize + gridSize / 2 - _offsetY * gridSize
                );
            }
        }
        _eventCTX.stroke();
        _eventCTX.beginPath();
        _eventCTX.fillStyle = _mapStyles['attack-style'];
        for (i = 0; i < _mapData[0].length; i++) {
            if (_mapData[0][i][0] === 0) {
                id = _mapData[0][i][1];
                if (_mapData[1][id] === undefined) {
                    continue;
                }
                style = _mapStyles['group'][_mapData[1][id][2]];
                result = _getOriginGridCoordinate(
                    _mapData[1][id],
                    _mapLastData !== undefined ? _mapLastData[1][id] : undefined,
                    style
                );
                _eventCTX.rect(
                    _mapData[0][i][2] * gridSize + gridSize / 2 - _offsetX * gridSize - gridSize / 8,
                    _mapData[0][i][3] * gridSize + gridSize / 2 - _offsetY * gridSize - gridSize / 8,
                    gridSize / 4,
                    gridSize / 4
                );
            }
        }
        _eventCTX.fill();
    }
}

function _drawMiniMAP() {
    if ($('#magnet-settings-minimap').is(':checked')) {
        var imgData = _minimapCTX.createImageData(_mapStyles['minimap-width'], _mapStyles['minimap-height']);
        for (var i = 0, size = _mapData[4].length; i < size; i++) {
            imgData.data[i * 4]     = (_mapData[4][i] >> 24) & 255;
            imgData.data[i * 4 + 1] = (_mapData[4][i] >> 16) & 255;
            imgData.data[i * 4 + 2] = (_mapData[4][i] >> 8 ) & 255;
            imgData.data[i * 4 + 3] = _mapData[4][i]         & 255;
        }
        _minimapCTX.putImageData(imgData, 0, 0);
        _minimapCTX.strokeRect(0, 0, _mapStyles['minimap-width'], _mapStyles['minimap-height']);
    } else {
        _minimapCTX.clearRect(0, 0, _minimapCTX.canvas.width, _minimapCTX.canvas.height);
    }
}

function _drawMiniMAPPosition() {
    _minimapPostionCTX.clearRect(0, 0, _minimapPostionCTX.canvas.width, _minimapPostionCTX.canvas.height);
    if ($('#magnet-settings-minimap').is(':checked')) {
        var offsetXMini = _offsetX / _mapStyles['width'] * _minimapPostionCTX.canvas.width;
        var offsetYMini = _offsetY / _mapStyles['height'] * _minimapPostionCTX.canvas.height;
        var xLength = Math.min(_mapStyles['width'], _gridCTX.canvas.width / gridSize) / _mapStyles['width'] * _minimapPostionCTX.canvas.width;
        var yLength = Math.min(_mapStyles['height'], _gridCTX.canvas.height / gridSize) / _mapStyles['height'] * _minimapPostionCTX.canvas.height;

        _minimapPostionCTX.strokeStyle = "rgba(0,0,255,255)";
        _minimapPostionCTX.strokeRect(offsetXMini, offsetYMini, xLength, yLength);
    }
}

function _animate() {
    _mapProcessedFrame++;
    if ((_mapStatus !== 'STOP' && _mapStatus !== 'PLAY') || _mapForcedPause) { // Pause
        window.requestAnimationFrame(_animate);
    } else if (_mapStatus === 'PLAY' && _mapData !== undefined) {
        if (_isBrowserSizeChanged) {
            _gridCTX.canvas.width = window.innerWidth;
            _gridCTX.canvas.height = window.innerHeight;

            _agentRangeCTX.canvas.width = window.innerWidth;
            _agentRangeCTX.canvas.height = window.innerHeight;

            _eventCTX.canvas.width = window.innerWidth;
            _eventCTX.canvas.height = window.innerHeight;

            _agentBodyCTX.canvas.width = window.innerWidth;
            _agentBodyCTX.canvas.height = window.innerHeight;

            _agentHPCTX.canvas.width = window.innerWidth;
            _agentHPCTX.canvas.height = window.innerHeight;

            _staticCTX.canvas.width = window.innerWidth;
            _staticCTX.canvas.height = window.innerHeight;

            _isBrowserSizeChanged = false;
        }
        if (_isGridSizeChanged) {
            _drawGrid();
            _isGridSizeChanged = false;
        }
        if (_isWindowChanged) {
            _drawObstacles();
            _drawMiniMAPPosition();
            if (_mapAnimateTick === 0) {
                _isWindowChanged = false;
            }
        }
        if (_mapAnimateTick === 0) {
            _drawNumbers();
            _drawMiniMAP();
        }

        if (_mapAnimateTick < TOTAL_STEP) {
            _drawAgent(_mapData[1], _mapLastData !== undefined ? _mapLastData[1] : undefined);
            _drawEvent();
            _mapAnimateTick += _mapSpeed;
        } else {
            _mapProcessedImage++;
            _mapCurrentImage++;
            _mapAnimateTick = 0;
            if (_mapCurrentImage >= _mapTotalImage) {
                _mapCurrentImage = _mapTotalImage - 1;
                _socket.send('p' + _mapCurrentImage.toString() + ' '
                    + Math.floor(_offsetX).toString() + ' ' + Math.floor(_offsetY).toString() + ' '
                    + Math.ceil(_offsetX + window.innerWidth / gridSize).toString() + ' '
                    + Math.ceil(_offsetY + window.innerHeight / gridSize).toString());
                _mapStatus = 'WAITING';
            } else {
                $('#magnet-settings-progress').val(_mapCurrentImage);
                _socket.send('p' + _mapCurrentImage.toString() + ' '
                    + Math.floor(_offsetX).toString() + ' ' + Math.floor(_offsetY).toString() + ' '
                    + Math.ceil(_offsetX + window.innerWidth / gridSize).toString() + ' '
                    + Math.ceil(_offsetY + window.innerHeight / gridSize).toString());
                _mapStatus = 'WAITING';
            }
        }
        window.requestAnimationFrame(_animate);
    } else {
        _minimapCTX.clearRect(0, 0, _minimapCTX.canvas.width, _minimapCTX.canvas.height);
        _staticCTX.clearRect(0, 0, _staticCTX.canvas.width, _staticCTX.canvas.height);
        _gridCTX.clearRect(0, 0, _gridCTX.canvas.width, _gridCTX.canvas.height);
        _agentRangeCTX.clearRect(0, 0, _agentRangeCTX.canvas.width, _agentRangeCTX.canvas.height);
        _agentHPCTX.clearRect(0, 0, _agentHPCTX.canvas.width, _agentHPCTX.canvas.height);
        _agentBodyCTX.clearRect(0, 0, _agentBodyCTX.canvas.width, _agentBodyCTX.canvas.height);
        _eventCTX.clearRect(0, 0, _eventCTX.canvas.width, _eventCTX.canvas.height);
        _minimapPostionCTX.clearRect(0, 0, _minimapPostionCTX.canvas.width, _minimapPostionCTX.canvas.height);
    }
}
