from asgiref.wsgi import WsgiToAsgi
from app import StatefulJobAssessmentSystem, A2AServer

assessment_system = StatefulJobAssessmentSystem()
a2a_server = A2AServer(assessment_system)
app = WsgiToAsgi(a2a_server.app)