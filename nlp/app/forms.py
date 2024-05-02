from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class MyForm(FlaskForm):
	class Meta:  # Ignoring CSRF security feature.
		csrf = False

	input_field = StringField(label='Ask me anything about Tulane Majors, Minors, or Graduate Programs!\n', id='input_field',
							  validators=[DataRequired()], 
							  render_kw={'style': 'width:100%'})
	submit = SubmitField('Tell me!')