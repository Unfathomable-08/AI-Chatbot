const chatApp = () => {
    return {
        input: '',
        loading: false,
        messages: [],

        sentMessage(){
            if (this.input.trim() !== ""){
                console.log(this.input)
                this.loading = true;
                this.messages.push({ sender: 'you', text: this.input , _id: this.messages.length + 1});
                console.log(this.messages)
                this.input = ''
            }
        }
    }
}