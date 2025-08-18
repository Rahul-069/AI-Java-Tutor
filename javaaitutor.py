!pip install -q transformers accelerate gradio

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from typing import List, Dict, Any
from collections import defaultdict
import re
import random

class JavaTutorAI:
    def __init__(self):
        # Initialize the model
        print("Loading DeepSeek Coder model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).cuda()

        # Learning tracking
        self.conversation_history = []
        self.learning_progress = {
            'topics_covered': defaultdict(int),
            'questions_asked': 0,
            'code_examples_provided': 0,
            'session_start': datetime.datetime.now(),
            'daily_progress': defaultdict(list)
        }

        # Quiz tracking
        self.quiz_progress = {
            'total_quizzes_taken': 0,
            'quizzes_by_topic': defaultdict(int),
            'correct_answers': 0,
            'total_questions': 0,
            'quiz_history': [],
            'topic_scores': defaultdict(list),
            'daily_quiz_progress': defaultdict(list)
        }

        # Current quiz state
        self.current_quiz = None
        self.current_question_index = 0

        # Java topics for tracking
        self.java_topics = {
            'basics': ['variable', 'data type', 'string', 'array', 'loop', 'if', 'else'],
            'oop': ['class', 'object', 'inheritance', 'polymorphism', 'encapsulation', 'abstraction'],
            'advanced': ['exception', 'thread', 'collection', 'generics', 'lambda', 'stream'],
        }

        # Quiz questions database
        self.quiz_questions = {
            'basics': [
                {
                    'question': 'Which of the following is NOT a primitive data type in Java?',
                    'options': ['int', 'String', 'boolean', 'char'],
                    'correct': 1,
                    'explanation': 'String is a class in Java, not a primitive data type. The primitive data types are byte, short, int, long, float, double, boolean, and char.'
                },
                {
                    'question': 'What is the default value of an int variable in Java?',
                    'options': ['null', '0', '1', 'undefined'],
                    'correct': 1,
                    'explanation': 'The default value of an int variable in Java is 0.'
                },
                {
                    'question': 'Which loop is guaranteed to execute at least once?',
                    'options': ['for loop', 'while loop', 'do-while loop', 'enhanced for loop'],
                    'correct': 2,
                    'explanation': 'The do-while loop checks the condition after executing the body, so it always executes at least once.'
                },
                {
                    'question': 'What is the correct way to declare an array in Java?',
                    'options': ['int[] arr = new int[5];', 'int arr[] = new int[5];', 'Both A and B', 'array<int> arr = new array<int>(5);'],
                    'correct': 2,
                    'explanation': 'Both int[] arr = new int[5]; and int arr[] = new int[5]; are correct ways to declare an array in Java.'
                },
                {
                    'question': 'Which operator is used for string concatenation in Java?',
                    'options': ['&', '+', '||', '&&'],
                    'correct': 1,
                    'explanation': 'The + operator is used for string concatenation in Java.'
                }
            ],
            'oop': [
                {
                    'question': 'What is encapsulation in Java?',
                    'options': ['Creating multiple classes', 'Hiding implementation details', 'Creating objects', 'Inheriting properties'],
                    'correct': 1,
                    'explanation': 'Encapsulation is the concept of hiding the internal implementation details of a class and exposing only necessary information through public methods.'
                },
                {
                    'question': 'Which keyword is used to inherit a class in Java?',
                    'options': ['implements', 'extends', 'inherits', 'super'],
                    'correct': 1,
                    'explanation': 'The extends keyword is used to inherit a class in Java.'
                },
                {
                    'question': 'What is method overriding?',
                    'options': ['Creating multiple methods with same name', 'Redefining a method in subclass', 'Calling parent method', 'Creating abstract methods'],
                    'correct': 1,
                    'explanation': 'Method overriding is when a subclass provides a specific implementation of a method that is already defined in its parent class.'
                },
                {
                    'question': 'Which access modifier allows access from anywhere?',
                    'options': ['private', 'protected', 'public', 'default'],
                    'correct': 2,
                    'explanation': 'The public access modifier allows access from anywhere in the program.'
                },
                {
                    'question': 'What is the purpose of the super keyword?',
                    'options': ['Create new object', 'Access parent class members', 'Define static methods', 'Handle exceptions'],
                    'correct': 1,
                    'explanation': 'The super keyword is used to access parent class members (methods and variables) from a subclass.'
                }
            ],
            'advanced': [
                {
                    'question': 'Which exception is thrown when dividing by zero?',
                    'options': ['NullPointerException', 'ArithmeticException', 'NumberFormatException', 'ArrayIndexOutOfBoundsException'],
                    'correct': 1,
                    'explanation': 'ArithmeticException is thrown when an exceptional arithmetic condition occurs, such as dividing by zero.'
                },
                {
                    'question': 'What is the difference between ArrayList and LinkedList?',
                    'options': ['No difference', 'ArrayList is faster for random access', 'LinkedList is always better', 'ArrayList cannot store objects'],
                    'correct': 1,
                    'explanation': 'ArrayList is faster for random access operations because it uses an array internally, while LinkedList is better for frequent insertions and deletions.'
                },
                {
                    'question': 'Which collection does not allow duplicate elements?',
                    'options': ['List', 'Set', 'Queue', 'Map'],
                    'correct': 1,
                    'explanation': 'Set is a collection that does not allow duplicate elements.'
                },
                {
                    'question': 'What is a lambda expression in Java?',
                    'options': ['A type of loop', 'Anonymous function', 'Exception handling', 'Class definition'],
                    'correct': 1,
                    'explanation': 'A lambda expression is an anonymous function that can be used to create instances of functional interfaces.'
                },
                {
                    'question': 'Which keyword is used to create a thread in Java?',
                    'options': ['thread', 'new Thread()', 'start()', 'run()'],
                    'correct': 1,
                    'explanation': 'You create a thread using new Thread() constructor, then call start() to begin execution.'
                }
            ]
        }

    def detect_programming_language(self, user_input: str) -> str:
        """Detect if user is asking about a programming language other than Java"""
        user_text = user_input.lower()

        # Common programming languages (excluding Java)
        other_languages = {
            'python': ['python', 'py', 'django', 'flask', 'pandas', 'numpy', 'matplotlib'],
            'javascript': ['javascript', 'js', 'node.js', 'react', 'vue', 'angular', 'express'],
            'c++': ['c++', 'cpp', 'stl', 'iostream', '#include'],
            'c': ['c programming', 'printf', 'scanf', 'malloc', 'stdio.h'],
            'c#': ['c#', 'csharp', '.net', 'asp.net', 'unity'],
            'php': ['php', 'laravel', 'symfony', 'wordpress'],
            'ruby': ['ruby', 'rails', 'gem', 'bundler'],
            'go': ['golang', 'go lang', 'go programming'],
            'rust': ['rust', 'cargo', 'rustc'],
            'kotlin': ['kotlin', 'kotlinc'],
            'swift': ['swift', 'ios', 'xcode'],
            'dart': ['dart', 'flutter'],
            'scala': ['scala', 'sbt'],
            'r': [' r programming', 'rstudio', 'ggplot'],
            'matlab': ['matlab', 'simulink'],
            'perl': ['perl', 'cpan'],
            'html': ['html', 'css', 'html5', 'css3'],
            'sql': ['sql', 'mysql', 'postgresql', 'sqlite', 'oracle'],
            'bash': ['bash', 'shell script', 'terminal'],
            'powershell': ['powershell', 'ps1']
        }

        # Check for mentions of other languages
        for lang, keywords in other_languages.items():
            for keyword in keywords:
                if keyword in user_text:
                    return lang

        return 'java'  # Default to Java

    def generate_response(self, user_input: str, history: List[List[str]]) -> str:
        """Generate response using DeepSeek Coder model"""

        # First, check if user is asking about a different programming language
        detected_language = self.detect_programming_language(user_input)

        if detected_language != 'java':
            return f"""I appreciate your question about {detected_language.title()}! However, I'm specifically designed to be a Java tutor and my training is focused on Java programming concepts, syntax, and best practices.

I'd be happy to help you with:
✅ Java fundamentals (variables, loops, conditionals)
✅ Object-Oriented Programming in Java
✅ Java collections and data structures
✅ Exception handling in Java
✅ Java frameworks (Spring, Hibernate)
✅ Java best practices and coding standards

If you have any Java-related questions, please feel free to ask! For {detected_language.title()} programming, I'd recommend consulting resources specifically designed for that language."""

        # Create Java tutor system prompt
        system_prompt = """You are an expert Java tutor. Your goal is to help students learn Java programming effectively.

        Guidelines:
        - Provide clear, beginner-friendly explanations
        - Include practical code examples
        - Ask follow-up questions to ensure understanding
        - Break down complex concepts into simple steps
        - Encourage best practices and proper coding style
        - Be patient and supportive
        - ONLY discuss Java programming - politely redirect if asked about other languages

        Focus exclusively on Java programming concepts, syntax, and best practices."""

        # Prepare messages for the model
        messages = [
            {'role': 'system', 'content': system_prompt}
        ]

        # Add recent conversation history (ensure proper format)
        if history:
            for exchange in history[-3:]:  # Last 3 exchanges
                if len(exchange) >= 2:
                    human_msg, ai_msg = exchange[0], exchange[1]
                    if human_msg and ai_msg:
                        messages.append({'role': 'user', 'content': human_msg})
                        messages.append({'role': 'assistant', 'content': ai_msg})

        # Add current user input
        messages.append({'role': 'user', 'content': user_input})

        try:
            # Generate response
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id
            )

            response = self.tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            )

            # Track learning progress
            self.track_learning_progress(user_input, response)

            return response.strip()

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}. Please try again."

    def track_learning_progress(self, user_input: str, response: str):
        """Track learning progress and topics covered"""

        # Update counters
        self.learning_progress['questions_asked'] += 1

        # Check for code examples in response
        if '' in response or 'public class' in response or 'public static void main' in response:
            self.learning_progress['code_examples_provided'] += 1

        # Track topics mentioned
        user_text = user_input.lower()
        response_text = response.lower()
        combined_text = user_text + ' ' + response_text

        for category, topics in self.java_topics.items():
            for topic in topics:
                if topic in combined_text:
                    self.learning_progress['topics_covered'][topic] += 1

        # Daily progress tracking
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        self.learning_progress['daily_progress'][today].append({
            'timestamp': datetime.datetime.now().isoformat(),
            'question': user_input[:100],  # First 100 chars
            'topics_identified': [topic for category, topics in self.java_topics.items()
                                for topic in topics if topic in combined_text]
        })

        # Store conversation
        self.conversation_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'ai_response': response
        })

    def start_quiz(self, topic: str, num_questions: int = 5):
        """Start a new quiz"""
        if topic == "Overall":
            # Mix questions from all topics
            all_questions = []
            for topic_questions in self.quiz_questions.values():
                all_questions.extend(topic_questions)
            selected_questions = random.sample(all_questions, min(num_questions, len(all_questions)))
        else:
            topic_key = topic.lower()
            if topic_key in self.quiz_questions:
                available_questions = self.quiz_questions[topic_key]
                selected_questions = random.sample(available_questions, min(num_questions, len(available_questions)))
            else:
                return None, "Topic not found!"

        self.current_quiz = {
            'topic': topic,
            'questions': selected_questions,
            'current_question': 0,
            'score': 0,
            'answers': [],
            'start_time': datetime.datetime.now()
        }

        return self.get_current_question(), "Quiz started successfully!"

    def get_current_question(self):
        """Get the current question in the quiz"""
        if not self.current_quiz or self.current_quiz['current_question'] >= len(self.current_quiz['questions']):
            return None

        question_data = self.current_quiz['questions'][self.current_quiz['current_question']]
        question_num = self.current_quiz['current_question'] + 1
        total_questions = len(self.current_quiz['questions'])

        return {
            'question_number': question_num,
            'total_questions': total_questions,
            'question': question_data['question'],
            'options': question_data['options']
        }

    def submit_answer(self, selected_option: int):
        """Submit an answer and move to next question"""
        if not self.current_quiz:
            return "No active quiz!", None, False

        current_q = self.current_quiz['questions'][self.current_quiz['current_question']]
        is_correct = selected_option == current_q['correct']

        # Record the answer
        self.current_quiz['answers'].append({
            'question': current_q['question'],
            'selected': selected_option,
            'correct': current_q['correct'],
            'is_correct': is_correct,
            'explanation': current_q['explanation']
        })

        if is_correct:
            self.current_quiz['score'] += 1

        # Move to next question
        self.current_quiz['current_question'] += 1

        # Check if quiz is complete
        if self.current_quiz['current_question'] >= len(self.current_quiz['questions']):
            return self.finish_quiz(), None, True

        # Get next question
        next_question = self.get_current_question()
        feedback = f"{'✅ Correct!' if is_correct else '❌ Incorrect.'} {current_q['explanation']}"

        return feedback, next_question, False

    def finish_quiz(self):
        """Finish the current quiz and update progress"""
        if not self.current_quiz:
            return "No active quiz to finish!"

        # Calculate final score
        total_questions = len(self.current_quiz['questions'])
        score = self.current_quiz['score']
        percentage = (score / total_questions) * 100

        # Update quiz progress
        self.quiz_progress['total_quizzes_taken'] += 1
        self.quiz_progress['quizzes_by_topic'][self.current_quiz['topic']] += 1
        self.quiz_progress['correct_answers'] += score
        self.quiz_progress['total_questions'] += total_questions

        # Store topic score
        self.quiz_progress['topic_scores'][self.current_quiz['topic']].append({
            'score': score,
            'total': total_questions,
            'percentage': percentage,
            'date': datetime.datetime.now().isoformat()
        })

        # Store quiz in history
        quiz_record = {
            'topic': self.current_quiz['topic'],
            'score': score,
            'total': total_questions,
            'percentage': percentage,
            'duration': (datetime.datetime.now() - self.current_quiz['start_time']).total_seconds(),
            'answers': self.current_quiz['answers'],
            'date': datetime.datetime.now().isoformat()
        }

        self.quiz_progress['quiz_history'].append(quiz_record)

        # Daily progress
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        self.quiz_progress['daily_quiz_progress'][today].append(quiz_record)

        # Generate result summary
        result_html = f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                    border-radius: 10px; color: white; text-align: center;">
            <h2>🎉 Quiz Completed!</h2>
            <div style="font-size: 3em; margin: 20px 0;">
                {score}/{total_questions}
            </div>
            <div style="font-size: 2em; margin: 10px 0;">
                {percentage:.1f}%
            </div>
            <div style="margin: 20px 0;">
                <strong>Topic:</strong> {self.current_quiz['topic']}
            </div>
            <div style="margin: 20px 0;">
                <strong>Performance:</strong>
                {'🌟 Excellent!' if percentage >= 90 else
                '👍 Good Job!' if percentage >= 70 else
                '📚 Keep Learning!' if percentage >= 50 else
                '💪 Practice More!'}
            </div>
        </div>
        """

        # Store the final results for later display
        self.quiz_final_results = result_html

        # DON'T clear current quiz yet - we need it for the Results button
        # self.current_quiz = None  # Remove this line

        return result_html

    def get_quiz_dashboard(self):
        """Generate quiz progress dashboard"""
        total_quizzes = self.quiz_progress['total_quizzes_taken']
        total_questions = self.quiz_progress['total_questions']
        correct_answers = self.quiz_progress['correct_answers']
        overall_accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0

        dashboard_html = f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
                    border-radius: 10px; color: white;">
            <h2 style="text-align: center; margin-bottom: 30px;">🏆 Quiz Progress Dashboard</h2>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px;">
                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #FFD700;">📊 Total Quizzes</h3>
                    <p style="font-size: 2em; margin: 10px 0; font-weight: bold;">{total_quizzes}</p>
                </div>

                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #FFD700;">❓ Questions Answered</h3>
                    <p style="font-size: 2em; margin: 10px 0; font-weight: bold;">{total_questions}</p>
                </div>

                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #FFD700;">✅ Correct Answers</h3>
                    <p style="font-size: 2em; margin: 10px 0; font-weight: bold;">{correct_answers}</p>
                </div>

                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #FFD700;">🎯 Overall Accuracy</h3>
                    <p style="font-size: 2em; margin: 10px 0; font-weight: bold;">{overall_accuracy:.1f}%</p>
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #FFD700;">📈 Quizzes by Topic</h3>
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
        """

        # Add topic statistics
        for topic, count in self.quiz_progress['quizzes_by_topic'].items():
            dashboard_html += f"""
                <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px;
                     font-size: 0.9em; white-space: nowrap;">
                    {topic.title()} ({count})
                </span>
            """

        dashboard_html += """
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px;">
                <h3 style="margin-top: 0; color: #FFD700;">🏅 Recent Quiz Results</h3>
                <div style="max-height: 200px; overflow-y: auto;">
        """

        # Add recent quiz results
        recent_quizzes = self.quiz_progress['quiz_history'][-5:]  # Last 5 quizzes
        for quiz in reversed(recent_quizzes):
            date = datetime.datetime.fromisoformat(quiz['date']).strftime('%Y-%m-%d %H:%M')
            dashboard_html += f"""
                <div style="background: rgba(255,255,255,0.1); padding: 10px; margin: 10px 0; border-radius: 5px;">
                    <strong>{quiz['topic']}</strong> - {quiz['score']}/{quiz['total']} ({quiz['percentage']:.1f}%) - {date}
                </div>
            """

        dashboard_html += """
                </div>
            </div>
        </div>
        """

        return dashboard_html

    def get_learning_dashboard(self):
        """Generate learning progress dashboard"""

        # Calculate session duration
        session_duration = datetime.datetime.now() - self.learning_progress['session_start']
        hours, remainder = divmod(session_duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)

        # Create dashboard HTML
        dashboard_html = f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
            <h2 style="text-align: center; margin-bottom: 30px;">🎓 Java Learning Dashboard</h2>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px;">
                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #ffd700;">📊 Questions Asked</h3>
                    <p style="font-size: 2em; margin: 10px 0; font-weight: bold;">{self.learning_progress['questions_asked']}</p>
                </div>

                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #ffd700;">💻 Code Examples</h3>
                    <p style="font-size: 2em; margin: 10px 0; font-weight: bold;">{self.learning_progress['code_examples_provided']}</p>
                </div>

                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #ffd700;">⏱ Session Time</h3>
                    <p style="font-size: 1.5em; margin: 10px 0; font-weight: bold;">{int(hours)}h {int(minutes)}m</p>
                </div>

                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #ffd700;">📚 Topics Covered</h3>
                    <p style="font-size: 2em; margin: 10px 0; font-weight: bold;">{len(self.learning_progress['topics_covered'])}</p>
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px;">
                <h3 style="margin-top: 0; color: #ffd700;">🎯 Most Discussed Topics</h3>
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
        """

        # Add top topics
        sorted_topics = sorted(self.learning_progress['topics_covered'].items(),
                             key=lambda x: x[1], reverse=True)[:10]

        for topic, count in sorted_topics:
            dashboard_html += f"""
                <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px;
                     font-size: 0.9em; white-space: nowrap;">
                    {topic.title()} ({count})
                </span>
            """

        dashboard_html += """
                </div>
            </div>
        </div>
        """

        return dashboard_html

# Initialize the tutor
tutor = JavaTutorAI()

def chat_interface(message, history):
    """Main chat interface function"""
    if not message.strip():
        return history

    # Get response from tutor
    response = tutor.generate_response(message, history)

    # Add the new exchange to history
    history.append([message, response])

    return history

def clear_chat():
    """Clear the chat history"""
    return []

def refresh_dashboard():
    """Refresh the learning dashboard"""
    return tutor.get_learning_dashboard()

def refresh_quiz_dashboard():
    """Refresh the quiz dashboard"""
    return tutor.get_quiz_dashboard()

def show_quiz_results():
    """Show final quiz results when Results button is clicked"""
    if hasattr(tutor, 'quiz_final_results'):
        final_results = tutor.quiz_final_results

        # Now clear the quiz since we're showing final results
        tutor.current_quiz = None

        return (
            final_results,  # Show the green box with final results
            gr.update(visible=False),  # Hide submit button
            gr.update(visible=False),  # Hide next button
            gr.update(visible=False),  # Hide results button
            "",  # Clear question
            gr.update(choices=[], value=None, label="")  # Clear options and remove label
        )
    return "No quiz results available", gr.update(), gr.update(), gr.update(), "", gr.update()

def start_quiz_interface(topic, num_questions):
    """Start a new quiz from the interface"""
    question_data, message = tutor.start_quiz(topic, int(num_questions))
    if question_data:
        return (
            gr.update(visible=True),  # Show quiz interface
            gr.update(visible=False),  # Hide start interface
            f"**Question {question_data['question_number']}/{question_data['total_questions']}**\n\n{question_data['question']}",
            gr.update(choices=question_data['options'], value=None),
            "",  # Clear feedback
            gr.update(visible=True),  # Show submit button
            gr.update(visible=False),  # Hide next button
            gr.update(visible=False)   # Hide results button
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            gr.update(choices=[], value=None),
            f"Error: {message}",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

def submit_quiz_answer(selected_option):
    """Submit an answer in the quiz"""
    if selected_option is None:
        # User hasn't selected an option - show warning message
        return (
            "⚠ Please select an option before submitting your answer!",  # Warning message
            gr.update(visible=True),   # Keep submit button visible
            gr.update(visible=False),  # Keep next button hidden
            gr.update(visible=False),  # Keep results button hidden
            gr.update(),  # Keep current question unchanged
            gr.update()   # Keep current options unchanged
        )

    # Get current question before submitting (since submit_answer moves to next)
    current_question = tutor.get_current_question()
    if current_question:
        # Store current question info before submitting
        current_q_text = f"**Question {current_question['question_number']}/{current_question['total_questions']}**\n\n{current_question['question']}"

        selected_index = current_question['options'].index(selected_option)
        feedback, next_question, is_complete = tutor.submit_answer(selected_index)

        if is_complete:
            # Quiz is finished - show only explanation and Results button
            # Get the last question's explanation only (not the full results)
            last_question = tutor.current_quiz['questions'][tutor.current_quiz['current_question'] - 1]
            explanation_feedback = f"{'✅ Correct!' if selected_index == last_question['correct'] else '❌ Incorrect.'} {last_question['explanation']}"

            return (
                explanation_feedback,  # Show ONLY explanation/feedback (not final results)
                gr.update(visible=False),  # Hide submit button
                gr.update(visible=False),  # Hide next button
                gr.update(visible=True),   # Show results button
                current_q_text,  # Keep last question visible
                gr.update(value=selected_option)  # Keep the selected option visible
            )
        else:
            # Show feedback but DON'T show next question yet
            return (
                feedback,  # Show explanation/feedback
                gr.update(visible=False),  # Hide submit button
                gr.update(visible=True),   # Show next button
                gr.update(visible=False),  # Hide results button
                current_q_text,  # Keep current question visible
                gr.update(value=selected_option)  # Keep the selected option visible
            )

    return "Error occurred", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "", gr.update(choices=[], value=None)

def next_question():
    """Move to next question or finish quiz"""
    current_question = tutor.get_current_question()
    if current_question:
        return (
            "",  # Clear feedback
            gr.update(visible=True),   # Show submit button
            gr.update(visible=False),  # Hide next button
            gr.update(visible=False),  # Hide results button
            f"**Question {current_question['question_number']}/{current_question['total_questions']}**\n\n{current_question['question']}",  # Show next question
            gr.update(choices=current_question['options'], value=None)  # Show new options and clear selection
        )
    else:
        # Quiz finished, reset to start screen
        return reset_quiz()

def reset_quiz():
    """Reset quiz interface to start screen"""
    return (
        gr.update(visible=False),  # Hide quiz interface
        gr.update(visible=True),   # Show start interface
        "",  # Clear question
        gr.update(choices=[], value=None),  # Clear options
        "",  # Clear feedback
        gr.update(visible=False),  # Hide submit button
        gr.update(visible=False),  # Hide next button
        gr.update(visible=False)   # Hide results button
    )

# Create Gradio interface
with gr.Blocks(
    title="AI Java Tutor",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="gray",
        font=gr.themes.GoogleFont("Inter")
    ),
    css="""
    /* Global styles */
    .gradio-container {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        margin: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }

    /* Header styling */
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 20px 20px 0 0;
        text-align: center;
        margin-bottom: 0;
    }

    .header-section h1 {
        font-size: 2.5em;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .header-section p {
        font-size: 1.1em;
        margin: 15px 0 0 0;
        opacity: 0.9;
    }

    /* Tab styling */
    .tab-nav {
        background: white;
        border-radius: 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    .tab-nav button {
        background: transparent;
        border: none;
        padding: 15px 25px;
        font-size: 1.1em;
        font-weight: 600;
        color: #667eea;
        border-radius: 10px 10px 0 0;
        margin-right: 5px;
        transition: all 0.3s ease;
    }

    .tab-nav button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
    }

    .tab-nav button.selected {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    /* Chat interface */
    .chat-container {
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 20px;
        overflow: hidden;
    }

    .chatbot {
        background: #f8f9ff;
        border: none;
        border-radius: 15px;
    }

    .chat-message {
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Input styling */
    .chat-input {
        border: 2px solid #e1e5f7;
        border-radius: 25px;
        padding: 15px 20px;
        font-size: 1.1em;
        transition: all 0.3s ease;
        background: white;
    }

    .chat-input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }

    /* Button styling */
    .primary-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 1.1em;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .primary-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    .secondary-btn {
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 1.1em;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .secondary-btn:hover {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
    }

    /* Dashboard cards */
    .dashboard-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }

    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }

    /* Quiz interface */
    .quiz-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }

    .quiz-question {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
        border-left: 5px solid #667eea;
    }

    .quiz-options {
        margin: 20px 0;
    }

    .quiz-options label {
        display: block;
        padding: 15px 20px;
        margin: 10px 0;
        background: white;
        border: 2px solid #e1e5f7;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .quiz-options label:hover {
        border-color: #667eea;
        background: #f8f9ff;
        transform: translateX(5px);
    }

    .quiz-options input[type="radio"]:checked + label {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }

    /* Success animations */
    .success-animation {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .header-section h1 {
            font-size: 2em;
        }

        .main-container {
            margin: 10px;
        }

        .dashboard-card {
            margin: 10px;
            padding: 20px;
        }

        .quiz-card {
            margin: 10px;
            padding: 20px;
        }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    """
) as app:

    # Header section
    with gr.Row():
        gr.HTML("""
        <div class="header-section">
            <h1>🤖 AI Java Tutor</h1>
            <p>Your intelligent companion for mastering Java programming</p>
        </div>
        """)

    with gr.Row():
        gr.HTML("""
        <div style="background: white; padding: 20px; margin: 0 20px; border-radius: 0 0 20px 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
            <p style="text-align: center; color: #667eea; font-size: 1.1em; margin: 0;">
                💡 <strong>Ask me anything about Java:</strong> Basic syntax • Object-Oriented Programming • Collections • Exception Handling
            </p>
        </div>
        """)

    with gr.Tab("💬 Chat", elem_classes="tab-nav"):
        with gr.Column(elem_classes="chat-container"):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                container=False,
                bubble_full_width=False,
                elem_classes="chatbot"
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="💭 Ask me anything about Java programming... (e.g., 'How do I create a class?' or 'Explain inheritance')",
                    container=False,
                    scale=4,
                    elem_classes="chat-input"
                )

            with gr.Row():
                with gr.Column(scale=1):
                    submit_btn = gr.Button("🚀 Send", variant="primary", elem_classes="primary-btn")
                with gr.Column(scale=1):
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", elem_classes="secondary-btn")

    with gr.Tab("📊 Learning Dashboard", elem_classes="tab-nav"):
        with gr.Column(elem_classes="dashboard-card"):
            dashboard_display = gr.HTML(value=tutor.get_learning_dashboard())

            with gr.Row():
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary", elem_classes="primary-btn")

    with gr.Tab("🧠 Quiz Center", elem_classes="tab-nav"):
        with gr.Row():
            with gr.Column(scale=1, elem_classes="quiz-card"):
                # Quiz start interface
                quiz_start_interface = gr.Group(visible=True)
                with quiz_start_interface:
                    gr.HTML("""
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h2 style="color: #667eea; font-size: 2em; margin: 0;">🎯 Test Your Knowledge</h2>
                        <p style="color: #666; font-size: 1.1em; margin: 10px 0;">Challenge yourself with interactive Java quizzes</p>
                    </div>
                    """)

                    topic_dropdown = gr.Dropdown(
                        choices=["Overall", "Basics", "OOP", "Advanced"],
                        value="Overall",
                        label="🎲 Select Topic",
                        elem_classes="quiz-dropdown"
                    )

                    num_questions_slider = gr.Slider(
                        minimum=3,
                        maximum=10,
                        value=5,
                        step=1,
                        label="📝 Number of Questions"
                    )

                    start_quiz_btn = gr.Button("🚀 Start Quiz", variant="primary", elem_classes="primary-btn", size="lg")

                # Quiz interface
                quiz_interface = gr.Group(visible=False)
                with quiz_interface:
                    question_display = gr.Markdown("", elem_classes="quiz-question")

                    answer_options = gr.Radio(
                        choices=[],
                        label="✅ Select your answer:",
                        value=None,
                        elem_classes="quiz-options"
                    )

                    with gr.Row():
                        submit_answer_btn = gr.Button("📤 Submit Answer", variant="primary", visible=False, elem_classes="primary-btn")
                        next_question_btn = gr.Button("➡️ Next Question", variant="secondary", visible=False, elem_classes="secondary-btn")
                        results_btn = gr.Button("🏆 View Results", variant="success", visible=False, elem_classes="success-animation")

                    feedback_display = gr.HTML("", label="💡 Feedback")

            with gr.Column(scale=1, elem_classes="quiz-card"):
                # Quiz dashboard
                gr.HTML("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="color: #667eea; font-size: 1.5em; margin: 0;">📈 Quiz Statistics</h3>
                </div>
                """)

                quiz_dashboard_display = gr.HTML(value=tutor.get_quiz_dashboard())

                with gr.Row():
                    with gr.Column(scale=1):
                        refresh_quiz_btn = gr.Button("🔄 Refresh Stats", variant="primary", elem_classes="primary-btn")
                    with gr.Column(scale=1):
                        reset_quiz_btn = gr.Button("🔄 Reset Quiz", variant="secondary", elem_classes="secondary-btn")

    with gr.Tab("📚 Java Reference", elem_classes="tab-nav"):
        with gr.Column(elem_classes="dashboard-card"):
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 30px;">
                <h2 style="color: #667eea; font-size: 2em; margin: 0;">📚 Java Quick Reference</h2>
                <p style="color: #666; font-size: 1.1em; margin: 10px 0;">Essential Java concepts and syntax at your fingertips</p>
            </div>
            """)

            with gr.Accordion("🔤 Data Types & Variables", open=False):
                gr.Markdown("""
                ```java
                // Primitive types
                int number = 42;
                double decimal = 3.14;
                boolean flag = true;
                char letter = 'A';

                // Reference types
                String text = "Hello World";
                int[] numbers = {1, 2, 3, 4, 5};
                ```
                """)

            with gr.Accordion("🔄 Control Structures", open=False):
                gr.Markdown("""
                ```java
                // If-else statement
                if (condition) {
                    // code block
                } else if (anotherCondition) {
                    // another code block
                } else {
                    // default code block
                }

                // Loops
                for (int i = 0; i < 10; i++) {
                    System.out.println(i);
                }

                while (condition) {
                    // loop body
                }

                do {
                    // loop body
                } while (condition);
                ```
                """)

            with gr.Accordion("🏗️ Classes & Objects", open=False):
                gr.Markdown("""
                ```java
                public class Student {
                    private String name;
                    private int age;

                    // Constructor
                    public Student(String name, int age) {
                        this.name = name;
                        this.age = age;
                    }

                    // Getter and Setter methods
                    public String getName() { return name; }
                    public void setName(String name) { this.name = name; }

                    public int getAge() { return age; }
                    public void setAge(int age) { this.age = age; }
                }
                ```
                """)

            with gr.Accordion("📦 Common Methods & Collections", open=False):
                gr.Markdown("""
                ```java
                // ArrayList
                ArrayList<String> list = new ArrayList<>();
                list.add("Java");
                list.add("Python");
                list.size();

                // HashMap
                HashMap<String, Integer> map = new HashMap<>();
                map.put("apple", 5);
                map.get("apple");

                // String operations
                String str = "Hello World";
                str.length();
                str.toUpperCase();
                str.substring(0, 5);
                str.contains("Hello");
                ```
                """)

    def submit_message(message, history):
        if message.strip():
            updated_history = chat_interface(message, history)
            return updated_history, ""
        return history, message

    # Chat events
    submit_btn.click(submit_message, [msg, chatbot], [chatbot, msg])
    msg.submit(submit_message, [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_chat, None, [chatbot])

    # Dashboard events
    refresh_btn.click(refresh_dashboard, None, [dashboard_display])

    # Quiz events
    start_quiz_btn.click(
        start_quiz_interface,
        [topic_dropdown, num_questions_slider],
        [quiz_interface, quiz_start_interface, question_display, answer_options, feedback_display, submit_answer_btn, next_question_btn, results_btn]
    )

    submit_answer_btn.click(
        submit_quiz_answer,
        [answer_options],
        [feedback_display, submit_answer_btn, next_question_btn, results_btn, question_display, answer_options]
    )

    next_question_btn.click(
        next_question,
        None,
        [feedback_display, submit_answer_btn, next_question_btn, results_btn, question_display, answer_options]
    )

    results_btn.click(
        show_quiz_results,
        None,
        [feedback_display, submit_answer_btn, next_question_btn, results_btn, question_display, answer_options]
    )

    refresh_quiz_btn.click(refresh_quiz_dashboard, None, [quiz_dashboard_display])

    reset_quiz_btn.click(
        reset_quiz,
        None,
        [quiz_interface, quiz_start_interface, question_display, answer_options, feedback_display, submit_answer_btn, next_question_btn, results_btn]
    )

# Launch the app
if __name__ == "__main__":
    print("Starting AI Java Tutor...")
    print("The interface will be available at the generated URL.")

    # Try different ports if 7860 is occupied
    import socket
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    try:
        app.launch(
            share=True,  # Creates a public URL for Colab
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            quiet=False
        )
    except OSError:
        # If port 7860 is busy, find a free port
        free_port = find_free_port()
        print(f"Port 7860 is busy, trying port {free_port}")
        app.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=free_port,
            show_error=True,
            quiet=False
        )
