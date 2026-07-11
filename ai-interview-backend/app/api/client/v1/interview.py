from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.client.deps import get_current_user
from app.db.session import get_db
from app.deps import get_interview_service
from app.models.user import User
from app.schemas.client.interview import InterviewStart, AnswerSubmit
from app.schemas.response import ApiResponse
from app.services.client.interview_service import InterviewService
from app.workflows.interview.service import submit_answer as submit_answer_to_graph

router = APIRouter()


@router.post("/start")
async def start_interview(
    data: InterviewStart,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    interview_service: InterviewService = Depends(get_interview_service),
):
    """开始新的 AI 面试会话"""
    result = await interview_service.start_interview(
        db=db,
        user_id=current_user.id,
        resume_id=data.resume_id,
        target_position=data.target_position,
        difficulty=data.difficulty,
        total_questions=data.total_questions,
    )
    return ApiResponse.success(data=result)


@router.post("/{interview_id}/answer")
async def submit_answer(
    interview_id: int,
    data: AnswerSubmit,
    stream: bool = Query(default=False, description="是否使用 SSE 流式返回"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """提交当前题目的回答（?stream=true 启动 SSE 流式评分）"""
    if not stream:
        async for result in submit_answer_to_graph(
            db=db,
            user_id=current_user.id,
            interview_id=interview_id,
            answer=data.answer,
            stream=False,
        ):
            return ApiResponse.success(data=result)

    async def event_generator():
        async for sse_str in submit_answer_to_graph(
            db=db,
            user_id=current_user.id,
            interview_id=interview_id,
            answer=data.answer,
            stream=True,
        ):
            yield sse_str

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{interview_id}/report")
async def get_report(
    interview_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    interview_service: InterviewService = Depends(get_interview_service),
):
    """获取面试评估报告"""
    result = await interview_service.get_report(
        db=db,
        user_id=current_user.id,
        interview_id=interview_id,
    )
    return ApiResponse.success(data=result)


@router.get("/{interview_id}/messages")
async def get_messages(
    interview_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    interview_service: InterviewService = Depends(get_interview_service),
):
    """获取面试的所有对话消息"""
    result = await interview_service.get_interview_messages(
        db=db,
        user_id=current_user.id,
        interview_id=interview_id,
    )
    return ApiResponse.success(data=result)


@router.get("")
async def get_interviews(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    interview_service: InterviewService = Depends(get_interview_service),
):
    """获取当前用户的所有面试记录"""
    result = await interview_service.get_interviews(
        db=db,
        user_id=current_user.id,
    )
    return ApiResponse.success(data=result)


@router.delete("/{interview_id}")
async def delete_interview(
    interview_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    interview_service: InterviewService = Depends(get_interview_service),
):
    """删除面试记录"""
    result = await interview_service.delete_interview(
        db=db,
        user_id=current_user.id,
        interview_id=interview_id,
    )
    return ApiResponse.success(data=result)
