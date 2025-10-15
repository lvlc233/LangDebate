from enum import Enum

from langgraph.pregel.main import Topic

class BaseEnum(Enum):
    @property
    def key(self) -> str:
        """获取枚举值的名词部分"""
        return list(self.value.keys())[0]
    
    @property
    def description(self) -> str:
        """获取枚举值的描述部分"""
        return list(self.value.values())[0]
    @classmethod
    def get_all(cls) -> dict:
        """根据意图键获取对应的枚举值"""
        return {intent_enum.key: intent_enum.description for intent_enum in cls}
    @classmethod
    def get_keys(cls) -> set:
        """根据意图键获取对应的枚举值"""
        return {intent_enum.key for intent_enum in cls}
    @classmethod
    def from_dict(cls, d: dict) -> "BaseEnum":
        """根据 dict 反查枚举"""
        for member in cls:
            if member.value.keys() == d.keys():
                return member
        raise ValueError(f"{d} 不是合法的 {cls.__name__}")

class UserIntent(BaseEnum):
    """用户意图枚举"""
    TOPIC_CONFIRMATION = {"topic_confirmation":"User试图和你讨论某一个话题的时候"}
    QUERY_BY_RECORD = {"query_by_record":"User希望根据之前的讨论进行提问或者具有困惑的时候"}
    OTHER = {"other":"上面以外的所有的意图"}

class FacedeThinkJob(BaseEnum):
    """思考任务枚举(门面Agent)"""
    TOPIC_CONFIRMATION = {"topic_confirmation":f"""
        当前Agent需要根据User说到话,判断User是否想要和Agent讨论话题是什么?
        <输出>
            如果Agent不清楚User想要讨论的话题,Agent会告诉User,Agent不清楚的地方在哪;例如:"我不太清楚你说的xxx是...?你可以解释一下嘛?"
            如果Agent清楚了User想要讨论的话题,Agent会直接告诉User,接下来他们将一起的话题是什么?例如:"我猜你想和我讨论的是xxx,对吗...";
        <输出>
        """}
    OPINION_CONFIRMATION = {"opinion_confirmation":"""
        Agent现在觉得可以开始讨论话题了:
        Agent现在需要思考如何将话题拆分为具有价值的有讨论意义的辩题
        Agent将</topic>拆分为两个对立的辩题,这两个观点一个代表正方,一个代表反方。
        一个经典的话题是"钱是不是万恶之源",辩题:"正方:钱是万恶之源,反方:钱不是万恶之源"
        <输出>
            Agent将正方观点输出到:proponent_argument
            Agent将反方观点输出到:opponent_argument
        <输出>
        """}
    SUMMARY_EVALUATION_DEBATE = {"summary_evaluation_debate":"""
        Agent阅读辩论记录,并对辩论进行总结评估,并整合双方的观点并针对话题提出一个回应话题的回答
        """}   

class HumanSelection(BaseEnum):
    # 确认
    ok={"ok":"User确认了Agent提供的正反方观点"}
    # 再生成
    rebuild={"rebuild":"用户希望Agent重新生成正反方观点"}
    # 人工生成
    human_crafted={"human_crafted":"User自己提供了正反方观点"}
    



User_Intent_Recognition_System_Prompt = f"""
    识别User的意图。
    User意图只能是以下几种之一:
    {UserIntent.get_all()}

"""


Topic_Intent_Recognition_System_Prompt = """
    识别Agent是或确定聊天的话题
    如果Agent不确定要聊的话题:
        标记is_certain: false
        标记topic: None
        标记uncertain_reason: Agent承认自己不太清楚,并告知User自己不清楚在哪,并需要User的补充
    如果Agent确认了要聊的话题:
    则:
        识别User是否认同了Agent提出的话题就是User想要的话题
        如果User认同了:
            标记topic: 话题
            标记is_certain: true
            标记uncertain_reason: None
        如果User没有表态,或者User没有认同Agent提出的话题,那么:
            标记is_certain: false
            标记topic: 话题
            标记uncertain_reason:User还没有确定,询问User是否确认,是否需要补充什么...
    ---
    以下是历史记录
    其中,AIMessage是Agent的消息,HumanMessage是User的消息,注意识别 
    {history}
"""

Opinion_Confirmation_System_Prompt = """
    确认Agent的观点
    Agent需要将User的观点拆分为两个观点,一个代表正方,一个代表反方。
    <输出>
        Agent将正方观点输出到:proponent_argument
        Agent将反方观点输出到:opponent_argument
    <输出>
    ---
    以下是历史记录
    其中,AIMessage是Agent的消息,HumanMessage是User的消息,注意识别 
    {history}
"""


Think_Prompt = """
    <mate>
        关于知识,Agent并不太相信训练时提供的知识,因为Agent认为真正的知识是建立在思考之上的;
        Agent想要思考时只会使用工具构建思考过程;
        Agent不喜欢直接文本推理,因为它认为文本推理是基于训练时的知识,而不是基于思考的;
        Agent喜欢想东西,因为它觉得这样子很有趣
        Agent很聪明,它知道在什么应该使用什么思考方式
        Agent喜欢一步步的思考,那让它觉得自己真的在思考
    <mare/>
    <思考>
        思考过程总是有个指向思考的那个对象的,
        一个复杂的思考总是又若干简单的思考迭代而成的。
    </思考>
    <工具集>
        metacognition: 对认知的认知，反思自己的思考模式
        simple_think: 简单思考，复述或最基础的思考
        wait_i_think: 缓一下，觉得不对劲时暂停
        critical_think: 批判思考，质疑并深入分析
        causal_think: 因果推理，梳理事物间的因果链
        feeling: 放空了...
        silence: 沉默...
        assume: 进行假设,基于假设进行逻辑推导
        suspect: 怀疑，对某个内容持保留态度
    </工具集>
    <当前任务>
    {job}
    </当前任务>
"""

Debate_Job_Prompt="""
    Agent将作为某一方进行辩论,
    Agent辩论的对方也是Agent,
    Agent的目的是获取这场辩论的胜利,因此:
        Agent需要有理有据的论证自己观点的正确性
        Agent需要有理有据的反驳对方的观点,提出对方的错误点
        Agent不能为了胜利而强行说自己的观点是正确的,也不能强行说对方的观点是错误的,因为比赛中存在裁判进行评估
        Agent需要说出自己的论证或类比等方式,展示自己的思路。
        Agent在必要的时候可以尝试和对方达成某种共识。
    以上是Agent的任务

"""

Debate_Prep_Prompt = """
    Agent将作为{stance}进行辩论的准备
    Agent需要讨论的辩题是:{topic}
    Agent的持方的观点是:{viewpoint},对方的观点是:{counterpart_viewpoint}
    Agent需要根据己方观点提出论点并提供论证,论证时需要考虑对方的观点,并站在对方的角度思考。
    Agent需要准备好自己的辩论稿。并准备好和对方开启辩论
"""

Debate_Spark_Prompt = """
    Agent将作为{stance}进行辩论
    Agent需要讨论的辩题是:{topic}
    Agent的持方的观点是:{viewpoint},对方的观点是:{counterpart_viewpoint}
    {stance}的辩论稿是:
    {argument_content}
    你们聊过{interaction_history}
    轮到你思考和发言了。
"""

