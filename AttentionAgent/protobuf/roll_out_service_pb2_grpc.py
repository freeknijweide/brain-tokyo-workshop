# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from protobuf import roll_out_service_pb2 as protobuf_dot_roll__out__service__pb2


class RollOutServiceStub(object):
  """RPC service for roll-outs.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.performRollOut = channel.unary_unary(
        '/evolution_algorithms.RollOutService/performRollOut',
        request_serializer=protobuf_dot_roll__out__service__pb2.RollOutRequest.SerializeToString,
        response_deserializer=protobuf_dot_roll__out__service__pb2.RollOutResponse.FromString,
        )


class RollOutServiceServicer(object):
  """RPC service for roll-outs.
  """

  def performRollOut(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_RollOutServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'performRollOut': grpc.unary_unary_rpc_method_handler(
          servicer.performRollOut,
          request_deserializer=protobuf_dot_roll__out__service__pb2.RollOutRequest.FromString,
          response_serializer=protobuf_dot_roll__out__service__pb2.RollOutResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'evolution_algorithms.RollOutService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class ParameterSyncServiceStub(object):
  """RPC service for parameter synchronization.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.syncParameter = channel.unary_unary(
        '/evolution_algorithms.ParameterSyncService/syncParameter',
        request_serializer=protobuf_dot_roll__out__service__pb2.ParamSyncRequest.SerializeToString,
        response_deserializer=protobuf_dot_roll__out__service__pb2.ParamSyncResponse.FromString,
        )


class ParameterSyncServiceServicer(object):
  """RPC service for parameter synchronization.
  """

  def syncParameter(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ParameterSyncServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'syncParameter': grpc.unary_unary_rpc_method_handler(
          servicer.syncParameter,
          request_deserializer=protobuf_dot_roll__out__service__pb2.ParamSyncRequest.FromString,
          response_serializer=protobuf_dot_roll__out__service__pb2.ParamSyncResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'evolution_algorithms.ParameterSyncService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))