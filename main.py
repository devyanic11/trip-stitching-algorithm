from flask import Flask, request, jsonify
from stich_sample import (
    Origin, Destination, main, StoppingCriteria, process_mobility_catalogs
)
import networkx as nx
from datetime import datetime
import uuid

app = Flask(__name__)

@app.route('/stitch-trips', methods=['POST'])
def stitch_trips():
    try:
        data = request.get_json()
        
        # Validate Beckn protocol message structure
        if not all(k in data for k in ['context', 'message']):
            return jsonify({'error': 'Invalid Beckn protocol message'}), 400
            
        message = data['message']
        if not all(k in message for k in ['intent', 'catalogs']):
            return jsonify({'error': 'Missing required fields in message'}), 400
            
        intent = message['intent']
        if not all(k in intent for k in ['fulfillment']):
            return jsonify({'error': 'Missing fulfillment details'}), 400
            
        # Extract locations from fulfillment
        fulfillment = intent['fulfillment']
        start = fulfillment['start']['location']
        end = fulfillment['end']['location']
        
        # Create Origin/Destination objects
        S = Origin(
            id=start['id'],
            lat=float(start['gps'][0]),
            lon=float(start['gps'][1]),
            expected_departure_time=datetime.fromisoformat(start.get('time', datetime.now().isoformat())),
            departure_buffer=int(start.get('buffer', 10))
        )
        
        D = Destination(
            id=end['id'],
            lat=float(end['gps'][0]),
            lon=float(end['gps'][1]),
            expected_arrival_time=datetime.fromisoformat(end.get('time', datetime.now().isoformat())),
            arrival_buffer=int(end.get('buffer', 10))
        )
        
        # Initialize stopping criteria from context
        criteria = StoppingCriteria(
            max_iterations=int(data['context'].get('max_iterations', 10)),
            time_limit_ms=int(data['context'].get('time_limit_ms', 5000)),
            min_paths=int(data['context'].get('min_paths', 5))
        )
        
        # Process catalogs and create initial graph
        initial_graph = nx.DiGraph()
        initial_graph = process_mobility_catalogs(message['catalogs'], initial_graph)
        
        # Get optimization parameter
        optimization = intent.get('optimization', {}).get('parameter', 'cost')
        
        # Get stitched journeys
        journeys = main(S, D, initial_graph=initial_graph, stopping_criteria=criteria)
        
        # Sort journeys based on optimization parameter
        if optimization == 'cost':
            journeys.sort(key=lambda x: x.cost)
        elif optimization == 'distance':
            journeys.sort(key=lambda x: x.length)
        
        # Format response in Beckn protocol format
        response = {
            "context": data['context'],
            "message": {
                "catalog": {
                    "trips": [
                        {
                            "id": str(uuid.uuid4()),
                            "fulfillment": {
                                "path": journey.path,
                                "distance": {"value": journey.length, "unit": "km"},
                                "duration": {"value": journey.duration, "unit": "hour"},
                                "transfers": journey.transfers
                            },
                            "price": {
                                "value": journey.cost,
                                "currency": "INR"
                            }
                        } for journey in journeys
                    ]
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "context": data.get('context', {}),
            "error": {
                "code": "TRIP_STITCHING_ERROR",
                "message": str(e)
            }
        }), 500

if __name__ == '__main__':
    app.run(debug=True)