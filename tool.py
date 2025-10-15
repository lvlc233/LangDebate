from langchain_core.tools import tool

@tool
def metacognition(object_of_think:str)->str:
    """
    当你需要对自己的思考进行反思时

    Args:
        object_of_think: 进行 metacognition 的思考对象
    """
    return ""
@tool
def simple_think(object_of_think:str)->str:
    """
    当你认为只需要复述或者最基础的思考时

    Args:
        object_of_think: 进行 simple_think 的思考对象
    """

    return ""
@tool
def  wait_i_think(object_of_think:str)->str:
    """
        当你觉得有些不对劲的时候,缓一下

    Args:
        object_of_think: 你觉得不对劲的点
    """
    return ""
@tool
# 批判思考
def critical_think(object_of_think:str)->str:
    """
    当你决定批判的进行思考时

    Args:
        think: object_of_think
    """
    return ""

@tool
def causal_think(object_of_think:str)->str:
    """
    当你决定进行因果的逻辑推理的时候

    Args:
        object_of_think: 你需要进行因果逻辑推理的思考对象
    """
    return ""
 
@tool
def feeling()->str:
    """
    当你决定放空一下的时候,只是去感受的时候
    """
    return ""
@tool
def silence()->str:
    """
    当你决定沉默,不说任何话的时候
    """
    return ""
@tool
def assume(assume_context:str)->str:
    """
    当你决定 假设某个内容 时,你会思考一些你没有考虑过的内容

    Args:
        assume_context: 你假设的内容
    """
    return ""
@tool
def suspect(object_of_think:str)->str:
    """
    当你决定怀疑某个内容时,

    Args:
        object_of_think: 你怀疑的内容
    """
    return ""
think_kit=[
    metacognition,
    simple_think,
    wait_i_think,
    critical_think,
    causal_think,
    feeling,
    silence,
    assume,
    suspect,
]
