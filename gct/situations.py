# Copyright 2026 HooRii Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Situation nodes of the schema layer."""

SITUATIONS = [
    "dining out, restaurant, ordering food or dessert / 外出就餐、点菜、甜品",
    "cooking at home, groceries, meal planning / 在家做饭、买菜、安排饮食",
    "gift giving, celebration, birthday / 送礼、庆祝、生日",
    "travel planning, trip, hotel, flight / 出行计划、旅行、酒店、航班",
    "daily commute, driving, pickup and drop-off / 日常通勤、开车、接送",
    "vehicle use, parking, car maintenance / 用车、停车、车辆保养",
    "medical visit, doctor appointment, follow-up / 就医、复诊、医生预约",
    "medication, prescriptions, taking pills / 用药、处方、服药",
    "allergy and food safety risk / 过敏与食品安全风险",
    "illness, symptoms, current health state / 生病、症状、当前身体状态",
    "fitness, sports, exercise, outdoor activity / 健身、运动、户外活动",
    "child school, classes, homework, teachers / 孩子的学校、课程、作业、老师",
    "school events, parent meetings, performances / 学校活动、家长会、演出",
    "childcare, babysitting, who looks after the kids / 照看孩子、临时托管",
    "elder care, aging parents, caregivers / 老人照护、父母、护工",
    "family gathering, visiting relatives, holidays / 家庭聚会、走亲访友、节日",
    "pets, vet, pet care and boarding / 宠物、兽医、寄养",
    "work meetings, scheduling, calendar / 工作会议、排期、日程",
    "work projects, deadlines, deliverables / 工作项目、截止日期、交付",
    "coworkers, manager, team, reporting lines / 同事、上级、团队、汇报关系",
    "confidential work documents, sharing, approvals / 工作机密文件、分享、审批",
    "business trip, remote work, time zones / 出差、远程办公、时区",
    "household finance, budget, bills, payments / 家庭财务、预算、账单、付款",
    "subscriptions, memberships, renewals / 订阅、会员、续费",
    "shopping, orders, deliveries, returns / 购物、订单、送货、退换",
    "devices, accounts, passwords, access recovery / 设备、账户、密码、找回访问",
    "home, apartment, landlord, repairs, utilities / 住房、房东、维修、水电",
    "moving, address changes, relocation / 搬家、地址变更、迁居",
    "appointments and reservations, booking, cancellation / 预约与预订、取消",
    "rescheduled or changed plans, latest version / 改期或变更的安排、最新版本",
    "upcoming events this week or weekend / 本周或周末的活动",
    "past events, what already happened / 已经发生过的事",
    "contacting someone, phone, message, email / 联系某人、电话、消息、邮件",
    "identity, birthday, age, documents, passport / 身份、生日、年龄、证件、护照",
    "addresses, locations, opening hours, directions / 地址、地点、营业时间、路线",
    "preferences and habits, likes and dislikes / 偏好与习惯、喜欢与不喜欢",
    "temporary state versus long-term preference / 临时状态与长期偏好",
    "hobbies, entertainment, media, games / 爱好、娱乐、影音、游戏",
    "social plans with friends, invitations / 和朋友的安排、邀请",
    "neighbors, community, building / 邻居、社区、楼栋",
    "legal matters, policies, permissions, consent / 法律事务、政策、授权、同意",
    "emergency, urgent help, safety / 紧急情况、求助、安全",
    "weather, season, clothing / 天气、季节、穿着",
    "insurance, claims, coverage / 保险、理赔、保障范围",
    "education and courses for adults, learning / 成人课程、学习",
    "same name or who exactly is meant, disambiguation / 同名或指代谁、消歧",
    "rumor, unconfirmed information, pending confirmation / 传闻、未确认信息、待确认",
    "general question or small talk, no personal context / 一般问题或闲聊，不需要个人信息",
]
N_LATENT = 16
