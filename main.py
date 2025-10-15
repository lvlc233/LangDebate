"""
基于LangGraph 0.6.6的多Agent辩论系统
包含门面Agent和两个基础Agent进行自博弈辩论
"""
from langchain_core.callbacks import BaseCallbackHandler

from colorama import Fore, Style, init
from langchain_core.callbacks import BaseCallbackHandler

from langgraph.pregel.main import Topic
from tool import think_kit
from prompt import (
        Think_Prompt,

        UserIntent,
        User_Intent_Recognition_System_Prompt,
        
        FacedeThinkJob,
        Topic_Intent_Recognition_System_Prompt,
        
        HumanSelection,
        Opinion_Confirmation_System_Prompt,
        
        Debate_Job_Prompt,
        Debate_Prep_Prompt,
        Debate_Spark_Prompt,
         
    )

import os
import dotenv
import colorama
from colorama import Fore
from typing import Annotated,Optional,List,Union
from langchain_core.runnables import RunnableLambda

from langgraph.graph import StateGraph, END,MessagesState,add_messages, state
from langgraph.types import Command,interrupt
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    )
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
dotenv.load_dotenv()
# # # 确保 LangSmith 追踪启用--->只有这么做了才会开始追踪
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGSMITH_PROJECT')
# 在init_chat方法中初始化
colorama.init(autoreset=True)



init(autoreset=True)

class ToolCallback(BaseCallbackHandler):
    def __init__(self, color: str = "cyan"):
        super().__init__()
        # 支持 16 种颜色名 → Fore 常量
        self._color = getattr(Fore, color.upper(), Fore.CYAN)

    # 统一封装：亮颜色 + 高亮名称 + 复位
    def _print(self, tag: str, name: str = "", body: str = ""):
        print(f"{self._color}{Style.BRIGHT}{tag}"
              f"{Style.RESET_ALL}{self._color}{name}{body}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._print("【Tool】", f"{serialized['name']} | 入参: ", input_str)

    def on_tool_end(self, output, **kwargs):
        self._print("【Tool】返回: ", body=output)


def messages_to_str_list(messages:list[AnyMessage])->list[str]:
    """ 将消息列表转换为字符串 """
    return [f"{type(message).__name__}:{message.content}\n" for message in messages]

class FacadeAgent:
    """门面Agent类"""
    # ------ 初始化 -------
    def __init__(self, model_name: str = os.getenv("FACAD_MODEL_NAME")):
        self.model_name = model_name
        self.llm = ChatOpenAI(model_name=model_name)
    
    class FacadeState(BaseModel):
        facade_messages: Annotated[list[AnyMessage], add_messages]
        topic: Optional[str] = Field(None,description="辩论主题")
        proponent_viewpoint: Optional[str] = Field(None,description="正方观点")
        opponent_viewpoint: Optional[str] = Field(None,description="反方观点")
        speaker: Optional[str] = Field(None,description="当前发言者")
        turn: int = Field(0,description="当前轮次")
        interaction_history: Annotated[list[AnyMessage], add_messages] = Field(None,description="辩论交互历史")
    def init_chat(self,state:MessagesState)->FacadeState:
        """ 初始化聊天状态"""
        # 提取用户输入
        user_input = HumanMessage(content=state["messages"][-1].content)
        return {"facade_messages":[user_input]}
    
    # ------ 思考 -------
    class ThinkState(BaseModel):
        job: FacedeThinkJob = Field(description="思考来源")
        think_messages: Annotated[list[AnyMessage], add_messages]
    def think(self, state: ThinkState):
        """ 思考 """
        print(f"{Fore.BLACK}模型思考中,思考内容为: {state.job.key}")
        think_time=create_react_agent(
            model=self.llm,
            tools=think_kit,
            prompt=Think_Prompt.format(job=state.job.description),
        )
        think_result=think_time.invoke({"messages": state.think_messages},
        )
        return {"think_messages":think_result["messages"][-1]}

    # ------- 意图识别类型 ---------
    class UserIntentType(BaseModel):
        intent: str = Field(description=f"用户意图，必须是{UserIntent.get_keys()}中的一个") 
    def user_intent_recognition(self, state: FacadeState) -> Command[str]:
        """ 意图识别 """
        # 调用LLM进行意图识别
        intent_response = self.llm.with_structured_output(self.UserIntentType).invoke(
            [SystemMessage(content=User_Intent_Recognition_System_Prompt), 
            *state.facade_messages],
            config={"callbacks": [ToolCallback("cyan")]}
        )
        # 获取对应的枚举值并进行路由跳转
        intent = intent_response.intent
        # 在intent_recognition方法中
        print(f"{Fore.CYAN}用户意图为: {intent}")
        if intent in UserIntent.get_keys():
            return Command(
                goto=intent,
            )
        return Command(
            goto=END,
            update={"messages":AIMessage(content="我不理解你的意图")}
        )
    
    # ------ 话题确认 -------
    class TopicIntentType(BaseModel):
        is_certain:bool=Field(description=f"用户是否确认了话题")
        topic: str = Field(description=f"用户确认的话题")
        uncertain_reason: str = Field(description=f"不能未确认话题的原因")
    def topic_confirmation(self, state: FacadeState) -> Command[str]:
        """ 话题确认 """    
        # 思考用户的需求
        history = state.facade_messages
        state= self.ThinkState(job=FacedeThinkJob.TOPIC_CONFIRMATION,think_messages=history)
        think_result= RunnableLambda(self.think).invoke(state)
        history.append(think_result["think_messages"])
        # 识别进度
        topic_confirmation_response = self.llm.with_structured_output(self.TopicIntentType).invoke(
            [SystemMessage(content=Topic_Intent_Recognition_System_Prompt.format(
                history=messages_to_str_list(history))),],
            config={"callbacks": [ToolCallback("cyan")]}
        )

        # 是否已确认判断
        is_certain = topic_confirmation_response.is_certain
        if is_certain:
            print(f"{Fore.GREEN}话题已确认,结果为: {topic_confirmation_response.topic}")
            return Command(
                goto="opinion_confirmation",
                update={"messages":think_result["think_messages"],
                        "facade_messages":think_result["think_messages"],
                        "topic":topic_confirmation_response.topic,
                }
            )
        print(f"{Fore.GREEN}话题确认中...")
        return Command(
            goto=END,
            update={"messages":think_result["think_messages"],
                    "facade_messages":think_result["think_messages"],
            }
        )
    
    # ------ 观点确认 -------
    class OpinionStruct(BaseModel):
        proponent_viewpoint: str = Field(description=f"正方的观点")
        opponent_viewpoint: str = Field(description=f"反方的观点")

    def opinion_confirmation(self, state: FacadeState) ->FacadeState:
        """ 观点确认 """
        # 思考
        history = state.facade_messages
        history.append(SystemMessage(content=f"话题为: {state.topic}"))
        state= self.ThinkState(job=FacedeThinkJob.OPINION_CONFIRMATION,think_messages=history)
        think_result= RunnableLambda(self.think).invoke(state)
        history.append(think_result["think_messages"])
        # 结构化输出
        opinion_confirmation_response = self.llm.with_structured_output(self.OpinionStruct).invoke(
            [SystemMessage(content=Opinion_Confirmation_System_Prompt.format(   
                history=messages_to_str_list([think_result["think_messages"]]))),],
            config={"callbacks": [ToolCallback("cyan")]}
        )
        proponent_viewpoint=opinion_confirmation_response.proponent_viewpoint
        opponent_viewpoint=opinion_confirmation_response.opponent_viewpoint
        print(f"{Fore.BLUE}正方观点为: {proponent_viewpoint}")
        print(f"{Fore.RED}反方观点为: {opponent_viewpoint}")
        return {
            "messages":think_result["think_messages"],
            "facade_messages":think_result["think_messages"],
            "proponent_viewpoint":proponent_viewpoint,
            "opponent_viewpoint":opponent_viewpoint,
        }
    
        
    def human_approval(self, state: FacadeState) -> Command:
        """ 人工审核 """
        proponent_viewpoint=state.proponent_viewpoint
        opponent_viewpoint=state.opponent_viewpoint

          # 人工审核
        human_selection=interrupt(f"{Fore.BLUE}正方观点为: {proponent_viewpoint}\n{Fore.RED}反方观点为: {opponent_viewpoint}\n请审核")      
        if type(human_selection)==dict:
            human_selection=HumanSelection.from_dict(human_selection)
        if human_selection.key in HumanSelection.get_keys():
            if human_selection.key==HumanSelection.ok.key:
                return Command(
                    goto="debate_prep",
                    update={"messages":[SystemMessage(content="用户认同Agent拆分的正反方观点")],
                            "facade_messages":[SystemMessage(content="用户认同Agent拆分的正反方观点")],
                            "proponent_viewpoint":proponent_viewpoint,
                            "opponent_viewpoint":opponent_viewpoint,
                    }
                )
            if human_selection.key==HumanSelection.rebuild.key:
                return Command(
                    goto="opinion_confirmation",
                    update={"messages":[SystemMessage(content="用户未认同Agent拆分的正反方观点,用户需要Agent重新生成")],
                            "facade_messages":[SystemMessage(content="用户未认同Agent拆分的正反方观点,用户需要Agent重新生成")],
                    }
                )
            if human_selection.key==HumanSelection.human_crafted.key:
                human_crafted=interrupt("输入您的正反方观点:\n使用json的形式表达\n例如:{{\"proponent_viewpoint\":\"正方观点\",\"opponent_viewpoint\":\"反方观点\"}}")
                return Command(
                    goto="debate_prep",
                    update={"messages":[SystemMessage(content="用户输入了自定义的正反方观点")],
                            "facade_messages":[SystemMessage(content="用户输入了自定义的正反方观点")],
                            "proponent_viewpoint":human_crafted.proponent_viewpoint,
                            "opponent_viewpoint":human_crafted.opponent_viewpoint,
                    }
                )
        raise ValueError(f"human_selection key must be in {HumanSelection.get_keys()}")
    def debate_prep(self, state: FacadeState) -> List[Command[str]]:
        """ 辩论准备 """
        # 初始化辩论状态
        return [
            Command(
                goto="proponent_prep",
                update={"proponent_stance":"proponent",
                }
            ),
            Command(
                goto="opponent_prep",
                update={"opponent_stance":"opponent",
                }
            ),
            Command(
                update={"topic":state.topic,
                        "proponent_viewpoint":state.proponent_viewpoint,
                        "opponent_viewpoint":state.opponent_viewpoint,
                        "speaker":"proponent",
                        "turn":0,
                }
            ),
        ]
          
    def debate(self, state: FacadeState) -> Command[str]:
        """ 辩论准备 """
        if state.turn>=int(os.getenv("MAX_TURN")):
            return Command(
                goto="debate_end",
                update={"messages":state.interaction_history,
                        "facade_messages":state.interaction_history,
                }
            )
        if state.speaker=="proponent":
            return Command(
                goto="proponent_speak",
                update={"messages":state.interaction_history,
                        "facade_messages":state.interaction_history,
                        "speaker":"opponent",
                        "turn":state.turn,
                }
            )
        if state.speaker=="opponent":
            return Command(
                goto="opponent_speak",
                update={"messages":state.interaction_history,
                        "facade_messages":state.interaction_history,
                        "speaker":"proponent",
                        "turn":state.turn+1,
                }
            )

    def debate_end(self, state: FacadeState):
        """ 辩论结束 """
        # 总结辩论
        history = state.facade_messages
        state= self.ThinkState(job=FacedeThinkJob.SUMMARY_EVALUATION_DEBATE,think_messages=history)
        think_result= RunnableLambda(self.think).invoke(state)
        think_result=think_result["think_messages"]
        return {"messages":think_result,"facade_messages":think_result}
class DebateAgent:
    def __init__(self, model_name: str = os.getenv("DEBATE_MODEL_NAME")):
        self.model_name = model_name
        self.llm = ChatOpenAI(model_name=model_name)
        self.agent=self.get_debate_agent()
    # ------ 辩论智能体 ------
    def get_debate_agent(self) :
        return create_react_agent(
            model=self.llm,
            tools=think_kit,
            prompt=Think_Prompt.format(job=Debate_Job_Prompt),
        )

    class DebateState(BaseModel):
        """ 辩论状态 """
        topic: str = Field(description=f"辩论的主题")
        proponent_viewpoint: str = Field(description=f"正方的观点")
        opponent_viewpoint: str = Field(description=f"反方的观点")
        interaction_history: Annotated[list[AnyMessage], add_messages] = Field(description="交互历史")

        proponent_stance: str = Field(description=f"正方的立场")
        proponent_argument_content: Optional[str] = Field(default=None,description=f"正方的论证内容")
        proponent_think_messages: Annotated[list[AIMessage], add_messages] = Field(description=f"正方的思考过程")
        opponent_stance: str = Field(description=f"反方的立场")
        opponent_argument_content: Optional[str] = Field(default=None,description=f"反方的论证内容")
        opponent_think_messages: Annotated[list[AIMessage], add_messages] = Field(description=f"反方的思考过程")
    # ------ 辩论准备 ------
    # 我觉得这样子不好....
    def proponent_prep(self, state:DebateState)->DebateState:
        """ 正方辩论准备 """
        argument_content=self.agent.invoke({"messages":[SystemMessage(content=Debate_Prep_Prompt.format(
            stance=state.proponent_stance,
            topic=state.topic,
            viewpoint=state.proponent_viewpoint,
            counterpart_viewpoint=state.opponent_viewpoint,
        ))]},config={"callbacks": [ToolCallback("blue")]})
        argument_content=argument_content['messages'][-1].content
        print(f"{Fore.BLUE}正方辩论稿为: {argument_content}")
        return {"proponent_argument_content":argument_content}
    def opponent_prep(self, state:DebateState)->DebateState:
        """ 反方辩论准备 """
        argument_content=self.agent.invoke({"messages":[SystemMessage(content=Debate_Prep_Prompt.format(
            stance=state.opponent_stance,
            topic=state.topic,
            viewpoint=state.opponent_viewpoint,
            counterpart_viewpoint=state.proponent_viewpoint,
        ))]},config={"callbacks": [ToolCallback("red")]})
        argument_content=argument_content['messages'][-1].content
        print(f"{Fore.RED}反方辩论稿为: {argument_content}\n")
        return {"opponent_argument_content":argument_content}
    # ------ 对话 ------

    def proponent_speak(self, state:DebateState) -> DebateState:
        """ 辩论 """
        debate_response=self.agent.invoke({"messages":[SystemMessage(content=Debate_Spark_Prompt.format(
            stance=state.proponent_stance,
            topic=state.topic,
            viewpoint=state.proponent_viewpoint,
            counterpart_viewpoint=state.opponent_viewpoint,
            argument_content=state.proponent_argument_content,
            interaction_history=state.interaction_history,
        ))]},config={"callbacks": [ToolCallback("blue")]})
        spark=f"发言方:{state.proponent_stance}说"+debate_response['messages'][-1].content
        print(f"{Fore.BLUE}{spark}")
        return {"interaction_history":[AIMessage(content=spark)]}
    def opponent_speak(self, state:DebateState) -> DebateState:
        """ 辩论 """

        debate_response=self.agent.invoke({"messages":[SystemMessage(content=Debate_Spark_Prompt.format(
            stance=state.opponent_stance,
            topic=state.topic,
            viewpoint=state.opponent_viewpoint, 
            counterpart_viewpoint=state.proponent_viewpoint,
            argument_content=state.opponent_argument_content,
            interaction_history=state.interaction_history,
        ))]},config={"callbacks": [ToolCallback("red")]})
        
        spark=f"发言方:{state.opponent_stance}说"+debate_response['messages'][-1].content
        print(f"{Fore.RED}{spark}\n")
        return {"interaction_history":[AIMessage(content=spark)]}
def create_debate_graph() -> StateGraph:
    """创建辩论图"""
    
    # 创建图
    graph = StateGraph(MessagesState)
    
    # 添加节点
    facade_agent = FacadeAgent()
    proponent_agent = DebateAgent()
    opponent_agent = DebateAgent()

    # 初始化
    graph.add_node("init_chat", facade_agent.init_chat)
    # 识别用户意图
    graph.add_node("intent_recognition", facade_agent.user_intent_recognition)
    # 话题确认
    graph.add_node("topic_confirmation", facade_agent.topic_confirmation)
    # 辩题确认
    graph.add_node("opinion_confirmation", facade_agent.opinion_confirmation)
    graph.add_node("human_approval", facade_agent.human_approval)
    # 辩论搞准备
    graph.add_node("debate_prep", facade_agent.debate_prep)
    graph.add_node("proponent_prep", proponent_agent.proponent_prep) 
    graph.add_node("opponent_prep", opponent_agent.opponent_prep)
    # 辩论
    graph.add_node("debate", facade_agent.debate)
    graph.add_node("proponent_speak", proponent_agent.proponent_speak)
    graph.add_node("opponent_speak", opponent_agent.opponent_speak)
    # 对话
    graph.add_node("debate_end", facade_agent.debate_end)
    
    # 设置入口点
    graph.set_entry_point("init_chat")
    graph.add_edge("init_chat", "intent_recognition")
    graph.add_edge("opinion_confirmation", "human_approval")
    graph.add_edge(["proponent_prep","opponent_prep"], "debate")
    graph.add_edge("proponent_speak","debate")
    graph.add_edge("opponent_speak", "debate")
    graph.add_edge("debate_end",END)
    
    
    return graph

debate_graph=create_debate_graph()
if __name__ == "__main__":
    # 创建并运行辩论图
    debate_graph = create_debate_graph()

    # 编译图
    memory=MemorySaver()
    app = debate_graph.compile(checkpointer=memory)
    
    
    # 运行图
    final_state = app.invoke({"messages":[HumanMessage(content="我想和你讨论,llm是否可以仅凭借上下文做到学习?就这个为话题,无需向我再次确认,直接标记is_confirm=True")]},
        config={"configurable":{"thread_id":"1"}}
    )
    app.invoke(Command(resume=HumanSelection.ok),
        config={"configurable":{"thread_id":"1"}}
    )
    state=app.get_state( config={"configurable":{"thread_id":"1"}})
    from langchain_core.load import dumps
    print(dumps(state.values,ensure_ascii=False, indent=2))