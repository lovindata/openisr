import { BorderBox } from "../../atoms/BorderBox";

interface Props {
  src: string;
  name: string;
  width: number;
  height: number;
}

export function ImageCard({ src, name, width, height }: Props) {
  return (
    <BorderBox className="flex w-72 items-center justify-between p-3 text-xs">
      <div className="flex space-x-3">
        <img
          src={`http://localhost:8000${src}`} // TODO FIX endpoint issue because self-referencing on dev
          alt={name}
          className="h-12 w-12 rounded-lg"
        />
        <div className="flex flex-col justify-evenly">
          <label>{name}</label>
          <span>
            Source: {width}x{height}px
          </span>
        </div>
      </div>
      <div></div>
    </BorderBox>
  );
}
